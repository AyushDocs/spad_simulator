# SPAD Simulator

1D Poisson–Drift–Diffusion simulator for Separate Absorption, Grading, Charge-sheet, and Multiplication (SAGCM) InGaAs/InP Single-Photon Avalanche Diodes.

> 📖 **Full documentation:** [https://ayushdocs.github.io/spad_simulator/](https://ayushdocs.github.io/spad_simulator/)

## Physics

### Poisson Equation

Nonlinear 1D Poisson solved via Newton–Raphson with tridiagonal factorization:

```
d/dx [eps(x) dphi/dx] = -q[p(x) - n(x) + N_D(x) - N_A(x)]
```

Carrier statistics (Boltzmann): `n = n_i exp((phi - phi_n)/V_T)`, `p = n_i exp((phi_p - phi)/V_T)`.

Jacobian = tridiagonal with `dF_i/dphi_i = -(eps_{i+1/2} + eps_{i-1/2})/dx^2 - q(n+p)/V_T`. Solved via LAPACK banded solver. Voltage ramped 0→Vbias in 2V steps (previous solution seeds next guess). Damped Armijo line search, tolerance 5e-4.

### Impact Ionization (Van Overstraeten–de Man, default)

Two-regime Chynoweth form with optical-phonon temperature scaling (`T_ref = 300 K`):

```
alpha(E) = A_low  * exp(-B_low/E)   (E < E0)
         = A_high * exp(-B_high/E)  (E >= E0)

gamma(T) = tanh(hw/2kT_ref) / tanh(hw/2kT)
A(T) = A(T_ref) * gamma(T),  B(T) = B(T_ref) * gamma(T)
```

InP parameters at 300 K (`_facade.py:line-160`):

| Carrier | A_low | B_low | A_high | B_high | E0 | hw |
|---------|-------|-------|--------|--------|----|----|
| e⁻ | 1.12e7 | 3.11e6 | 2.93e6 | 2.64e6 | 3.85e5 | 0.063 eV |
| h⁺ | 4.79e6 | 2.55e6 | 1.62e6 | 2.11e6 | 3.85e5 | 0.063 eV |

Dead-space correction: `alpha_eff = alpha / (1 + alpha * l_d)`, `l_d = E_th / (q*|E|)`.

Okuto-Crowell model available as alternative (commented out in materials XML).

### Trigger Probability (McIntyre–Oldham–Hayat)

Coupled ODEs for electron/hole trigger probability, solved as integral fixed-point:

```
dPe/dx = -α(x)·(1-Pe)·(Pe+Ph-Pe·Ph)    Pe(W)=0
dPh/dx = +β(x)·(1-Ph)·(Pe+Ph-Pe·Ph)    Ph(0)=0
```

Integral form with dead-space delay (`l_e`, `l_h`):
```
Pe(x) = 1 - exp(-∫_x^W α(x')·Ptr(x'+l_e) dx')
Ph(x) = 1 - exp(-∫_0^x β(x')·Ptr(x'-l_h) dx')
```

Converges ~50 damped iterations (tolerance 1e-8).

### Multiplication Factor

McIntyre integral (electron injection). Two equivalent forms:

```
M = 1 / [1 - ∫ α·exp(∫ (β-α)) dx]   (_facade.py)
  = 1 / [1 - ∫ β·exp(∫ (α-β)) dx]   (breakdown.py)
```

Also computed via coupled ODE BVP (`MultiplicationSolver`):
```
dMn/dx = -α·(Mn+Mp),  dMp/dx = +β·(Mn+Mp),  Mn(W)=Mp(0)=1
```

Breakdown detected when M > 100.

### Dark Current & DCR

Three generation mechanisms:

| Component | Formula | Driver |
|-----------|---------|--------|
| SRH | `J = q·ni(T) / (2τ)` | Thermal, `ni(T) = √(Nc·Nv)·exp(-Eg(T)/2kT)` |
| BTBT | `A·F²·exp(-B·Eg^{3/2}/F)` | Kane tunneling at high field |
| TAT | Hurkx model | Trap-assisted tunneling |

DCR integrates generation × trigger probability:
```
DCR = A_det · ∫ [G_SRH + G_BTBT + G_TAT] · Ptr(x) dx
```
At 300 K, Vex=3 V: DCR ≈ 4,340 cps, BTBT dominant.

### PDP

```
PDP(λ) = (1-R) · T_dead(λ) · ∫ α_abs(λ)·exp(-α_abs·x) · Ptr(x) dx
```

### Carrier Transport

Field-dependent mobility (Caughey–Thomas):
```
mu(N,T) = [mu_min + (mu_max-mu_min)/(1+(N/N_ref)^γ)] · (T/300)^{-α_T}
v_d = mu·E / (1 + |E|/E_sat)
D = mu·V_T · (1 + 2·(|E|/v_sat)²)
```

### Self-Consistent PIC Loop

Couples carrier dynamics, Poisson, and quench circuit:
1. Deposit carrier charge on grid (CIC weighting) → `rho_ext`
2. Solve Poisson with `rho_ext` → updated E(x)
3. Drift-diffuse each carrier
4. Sample ionization: `P_ion = 1 - exp(-alpha_eff·dx)` after dead space
5. Spawn new e-h pairs on ionization
6. Circuit update: `V_spad = V_supply - I·R_q`
7. Repeat until V_spad < V_BR (quenched)

### Numerical Integration

All spatial integrals use `np.trapezoid` (composite trapezoidal rule, O(h²)).

## Device Structure

| # | Layer | Material | Thickness | Doping | Role |
|---|-------|----------|-----------|--------|------|
| 0 | Cap | InP | 2.5 µm | 2e18 cm⁻³ p+ | p-contact |
| 1 | Multiplication | InP | 0.5 µm | i | High-field avalanche |
| 2 | Charge sheet | InP | 0.2 µm | 1e17 cm⁻³ n+ | Field control |
| 3 | Grading | InGaAsP | 0.12 µm | i | Band offset smoothing |
| 4 | Absorber | InGaAs | 1.5 µm | i | Photon absorption |
| 5 | Buffer | InP | 0.5 µm | 5e16 cm⁻³ n+ | Transition |
| 6 | Substrate | InP | 2.0 µm | 1e18 cm⁻³ n+ | n-contact |

Breakdown voltage: ~60 V at 300 K (VODM). Typical excess bias: 2–5 V.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
make run         # CLI simulation
make run-ui      # GUI
```

Or programmatically:

```python
from src.main import build_sagcm_spad
from src.simulator import SPADSimulator

device = build_sagcm_spad()
sim = SPADSimulator(device)
Vbr, info = sim.find_breakdown(V_start=0, V_max=80, V_step=1.0)
print(f"Breakdown voltage: {Vbr:.1f} V")

# PDE spectrum
import numpy as np
wavelengths = np.linspace(900, 1700, 41) * 1e-9
pde = sim.compute_pde_spectrum(wavelengths, Vex=3.0)
```

## Running Tests

```bash
pytest
```

## Module Structure

```
src/
  core/            Grid, layers, materials, doping, device, constants
  poisson/         Nonlinear Poisson solver (Newton–Raphson), depletion width
  avalanche/       Impact ionization (VODM/OC), trigger (coupled ODEs),
                   multiplication (BVP), breakdown detection, dark current, PDP
  transport/       Drift-diffusion, Monte Carlo, timing jitter
  self_consistent/ PIC loop, particle-mesh (CIC), circuit quenching
  optimization/    PSO optimizer
  utils/           Exceptions, logging, XML loaders/ingestion, plotters
  main.py          Config, ingestion, artifacts, orchestration
  ui.py            GUI mode (tkinter)
```

### Architecture (src/main.py)

| Component | Class | Purpose |
|-----------|-------|---------|
| Data Ingestion Config | `DataIngestionConfig` | Paths to device/materials/absorption XML, sim params |
| Data Ingestion Service | `DataIngestionService` | Loads XML, builds `Device` and `SPADSimulator` |
| Simulation Artifact | `SimulationArtifact` | Structured container for all results |
| Artifact Writer | `ArtifactWriter` | Writes `SimulationArtifact` to XML |

### Plots Config

Enabled plots (`data/plots_config.xml`): `electric_field`, `trigger_profiles`, `atp`, `dcr_pde_vs_vex`, `dcr_vs_temp`, `iv_characteristic`, `multiplication_vs_vex`.

## Simulation Outputs

All plots saved to `plots/spad/`. 7 diagnostic plots + `sim_results.xml`.

### Electric Field

![Electric Field Profile](plots/spad/electric_field_profile.png)

**Electric Field E(x).** Field magnitude across device at Vex = 0–5 V. Peak field ~5.5×10⁵ V/cm in the thin InP multiplication layer. Charge sheet creates sharp drop, keeping absorber field < 10⁵ V/cm.

### Trigger Probability Profiles

![Trigger Probability](plots/spad/trigger_probability.png)

**Electron/Hole Trigger Probabilities Pe(x), Ph(x).** Pe peaks near p-contact, Ph near n-contact. Both → 1 at high Vex.

### Average Trigger Probability

![ATP vs Position](plots/spad/atp_vs_position.png)

**Average Trigger Probability Ptr(x) = Pe + Ph − Pe·Ph** across the device.

### I-V Characteristic

![IV Characteristic](plots/spad/iv_characteristic.png)

**Current–Voltage.** Dark and illuminated (1 µW, 1310 nm) curves. Sharp rise at V_BR marks Geiger-mode transition.

### DCR & PDE vs Excess Bias

![DCR & PDE vs Vex](plots/spad/dcr_pde_vs_vex.png)

**DCR and PDE vs Vex.** Both increase with Vex as trigger probability saturates.

### DCR vs Temperature

![DCR vs Temperature](plots/spad/dcr_vs_temperature.png)

**DCR vs T at Vex = 3 V.** Exponential rise via Varshni ni(T).

### Multiplication vs Excess Bias

![Multiplication vs Vex](plots/spad/multiplication_vs_vex.png)

**Avalanche multiplication factor M vs Vex.** M diverges at V_BR. M > 100 threshold defines breakdown.

### XML Artifact Output

```xml
<spad_simulation>
  <device>
    <breakdown_voltage_V>60.00</breakdown_voltage_V>
    <temperature_K>300.0</temperature_K>
    <detector_area_cm2>1.000000e-06</detector_area_cm2>
    <grid_N>500</grid_N>
    <total_thickness_cm>7.320000e-04</total_thickness_cm>
    <n_layers>7</n_layers>
  </device>
  <dark_current>
    <I_dark_A>2.800e-12</I_dark_A>
    <DCR_cps>4.340e+03</DCR_cps>
    <excess_voltage_V>3.0</excess_voltage_V>
  </dark_current>
  <pde_max>
    <PDE_1310nm wavelength="1310nm">0.68</PDE_1310nm>
  </pde_max>
</spad_simulation>
```

## Learning Notebooks

Six Jupyter notebooks in `notebooks/` walk through the physics step by step, with **interactive widgets** (sliders) for real-time parameter exploration:

| Notebook | Topic | Interactive |
|----------|-------|-------------|
| `01-AyushDocs-ElectricFieldSolver` | Poisson solver, Newton–Raphson, Boltzmann stats, band diagram | Bias slider for E-field & potential |
| `02-AyushDocs-IVandBreakdown` | McIntyre integral, breakdown detection, dark/light IV curves | Temperature slider for IV curve |
| `03-AyushDocs-IonizationCoefficients` | VODM vs Okuto-Crowell models, dead-space, temperature scaling | Field & temperature sliders |
| `04-AyushDocs-DarkCurrent` | SRH, BTBT, TAT generation, DCR vs temperature | Bias & temperature sliders |
| `05-AyushDocs-TriggerAndPDE` | Coupled McIntyre ODEs, trigger probability, PDP spectrum | Excess bias slider |
| `06-AyushDocs-ThreeRegionModel` | Constant-field closed-form, analytic McIntyre integral | Width & field sliders |

> **Note:** Interactive widgets require `ipywidgets` and `ipympl` (included with `pip install -e ".[dev]"`). Open each notebook in Jupyter Lab/Notebook and run all cells; sliders appear before the Summary section.

Run with the venv kernel (`spad-sim`), or execute headless:
```bash
cd notebooks/
jupyter nbconvert --to notebook --execute 01-AyushDocs-ElectricFieldSolver.ipynb --output /dev/null
```
> Headless execution will skip the interactive widget cells (no frontend to display them).

## License

MIT
