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

## Simulation Outputs

All plots are saved to `plots/spad/`. The simulator generates **15 diagnostic plots** and a structured JSON file summarising every metric.

### Device Structure

![Device Structure](plots/spad/device_structure.png)

**Figure 1 — Device Structure.** Top panel shows the material layers (InP, InGaAsP, InGaAs) along the growth direction. Bottom panel shows the net doping profile |N_D − N_A|. The p+ cap (2 × 10¹⁸ cm⁻³) and n+ substrate (1 × 10¹⁸ cm⁻³) sandwich the intrinsic multiplication and absorption regions. The charge sheet at 1 × 10¹⁷ cm⁻³ controls the electric field distribution between the multiplication and absorption zones.

### Electrostatic Potential

![Potential Profile](plots/spad/potential_profile.png)

**Figure 2 — Electrostatic Potential φ(x).** Potential profile at multiple bias voltages from 0 V to Vbr + 5 V. The potential drop is concentrated in the multiplication region (InP, 0.5 µm), confirming the high-field zone where impact ionization occurs. The grading layer smooths the heterojunction band offset between InP and InGaAs.

### Electric Field

![Electric Field Profile](plots/spad/electric_field_profile.png)

**Figure 3 — Electric Field E(x).** Electric field magnitude across the device at excess voltages Vex = 0–5 V. The peak field (~5.5 × 10⁵ V/cm) is localised in the thin InP multiplication layer. The field drops sharply in the absorber, ensuring carriers drift without ionising — critical for low-afterpulsing operation.

### Dark Current

![Dark Current vs Bias](plots/spad/dark_current_vs_bias.png)

**Figure 4 — Dark Current vs Excess Voltage.** Total dark current (I_dark) decomposed into its components: SRH thermal generation, band-to-band tunnelling (BTBT), and trap-assisted tunnelling (TAT). Thermal generation dominates below Vbr; tunnelling becomes significant at high fields near breakdown.

### Dark Count Rate

![DCR vs Bias](plots/spad/dcr_vs_bias.png)

**Figure 5 — Dark Count Rate vs Excess Voltage.** DCR = A_det × ∫ G(x) × P_trigger(x) dx, where G is the generation rate and P_trigger is the trigger probability. DCR increases with Vex because the trigger probability approaches unity in the high-field region.

### I-V Characteristic

![IV Characteristic](plots/spad/iv_characteristic.png)

**Figure 6 — Current–Voltage Characteristic.** Dark and illuminated I-V curves. The illuminated curve (1 µW at 1310 nm) shows the photocurrent contribution. The sharp current rise at Vbr marks avalanche breakdown. Below Vbr, the device operates as a conventional APD; above Vbr, it enters Geiger mode.

### PDP Spectrum

![PDP Spectrum](plots/spad/pdp_spectrum.png)

**Figure 7 — PDP vs Wavelength.** Photon detection probability across 900–1700 nm at multiple excess voltages. PDP peaks at ~74% near 1310 nm (InGaAs absorber bandgap) and drops at shorter wavelengths due to dead-zone absorption and at longer wavelengths due to sub-bandgap transparency. The PDP saturates for Vex ≥ 3 V.

### PDP vs Excess Voltage

![PDP vs Vex](plots/spad/pdp_vs_excess_voltage.png)

**Figure 8 — PDP vs Excess Voltage at Fixed Wavelengths.** PDP at 1100, 1310, 1550, and 1610 nm as a function of Vex. PDP increases with Vex as the trigger probability rises, then saturates when P_trigger ≈ 1. Shorter wavelengths show lower PDP due to absorption in the dead zone before the active absorber.

### Comprehensive I-V

![Comprehensive IV](plots/spad/comprehensive_iv.png)

**Figure 9 — Comprehensive I-V with Gain.** Left: dark current, primary photocurrent, and total illuminated current vs bias. Right: same with multiplication gain M overlaid. The gain rises steeply above Vbr, reaching M > 1000 for Vex > 5 V. The primary photocurrent is computed from Beer-Lambert absorption without multiplication.

### Trigger Probability Profiles

![Trigger Probability](plots/spad/trigger_probability.png)

**Figure 10 — Electron and Hole Trigger Probabilities Pe(x), Ph(x).** Spatial profiles of the electron-initiated (Pe, solid) and hole-initiated (Ph, dashed) trigger probabilities at Vex = 1, 3, and 5 V. Pe peaks near the p+ contact (electron injection side) and Ph peaks near the n+ contact. Both approach unity at high Vex, indicating self-sustaining avalanche.

### Afterpulsing vs Holdoff

![Afterpulsing](plots/spad/afterpulsing_vs_holdoff.png)

**Figure 11 — Afterpulsing Probability vs Holdoff Time.** P_ap(t_holdoff) = 1 − exp(−N_T τ_c (1 − exp(−t/τ_c))) for a single trap level with N_T = 10¹² cm⁻³ and τ_c = 1 µs. At short holdoff times, trapped carriers haven't been released, causing false afterpulse counts. At long holdoff, most traps have emptied. The optimal holdoff for 1% afterpulsing is annotated.

### Excess Noise Factor

![Excess Noise](plots/spad/excess_noise.png)

**Figure 12 — McIntyre Excess Noise Factor F(M).** F(M) = k_eff × M + (1 − k_eff) × (2 − 1/M), where k_eff = β/α is the effective ionisation rate ratio. Lower k_eff gives lower noise — InP's k_eff ≈ 0.56 indicates moderate noise. The shot-noise limit F = 1 is shown for reference.

### PDE vs Bias

![PDE vs Bias](plots/spad/pde_vs_bias.png)

**Figure 13 — Photon Detection Efficiency vs Excess Voltage.** PDE(1310 nm) as a function of Vex. PDE combines absorption probability, dead-zone transmission, and trigger probability. It saturates at ~68% when the trigger probability reaches unity.

### DCR vs Temperature

![DCR vs Temperature](plots/spad/dcr_vs_temperature.png)

**Figure 14 — Dark Count Rate vs Temperature.** DCR at Vex = 3 V measured at T = 285–315 K. DCR increases exponentially with temperature because ni(T) grows exponentially via the Varshni bandgap relation. The activation energy extracted from the slope identifies the dominant dark current mechanism (thermal generation vs tunnelling).

### Timing Jitter Histogram

![Timing Jitter](plots/spad/timing_jitter_histogram.png)

**Figure 15 — Single-Photon Timing Response (SPTR).** Histogram of detection times from Monte Carlo avalanche ensemble simulation. The FWHM and standard deviation σ characterise the timing resolution. Narrower histograms indicate better timing performance for LiDAR and time-correlated applications.

### JSON Metrics Output

A structured JSON file (`sim_results.json`) is written to the plots directory containing all computed metrics:

```json
{
  "device": {
    "Vbr_V": 75.0,
    "T_K": 300.0,
    "detector_area_cm2": 1e-06,
    "grid_N": 500,
    "grid_dx_cm": 1.467e-06,
    "total_thickness_cm": 0.000732,
    "n_layers": 7
  },
  "dark_current": {
    "I_dark_A": 3.05e-08,
    "DCR_cps": 1.95e9,
    "Vex_V": 3.0
  },
  "pdp_max": {
    "905nm": 0.143,
    "1310nm": 0.679,
    "1550nm": 0.401
  },
  "afterpulsing": {
    "N_T": 1e12,
    "tau_c": 1e-06,
    "P_ap_1us": 1.0,
    "holdoff_optimal_1pct_s": 1.01e-14
  },
  "excess_noise": {
    "M_max": 10000.0,
    "F_max": 5608.04,
    "k_eff": 0.561
  },
  "pde_1310nm": {
    "pde_max": 0.679,
    "wavelength_nm": 1310
  },
  "jitter": {
    "sigma_s": null,
    "fwhm_s": null
  },
  "dcr_vs_temperature": { ... },
  "pdp_vs_temperature": { ... }
}
```

## License

MIT
