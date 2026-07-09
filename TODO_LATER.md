# TODO Later — Physics Extensions

## High Priority

### Carrier Heating / Energy Balance
- Add energy balance equation alongside drift-diffusion
- Carrier temperature T_c > lattice T at high fields
- Field-dependent ionization coefficients depend on carrier energy, not just local field
- Improves breakdown voltage accuracy in thin multiplication regions

### Avalanche Buildup Time
- Finite multiplication rise time (~10-50 ps in InP)
- Affects timing resolution and gating behavior
- Model temporal evolution of carrier population during avalanche
- Extract RC time constant from quenching circuit coupling

### Optical Generation Profile
- Wavelength-dependent absorption coefficient alpha(lambda) per material
- Generate carrier injection profile: G(x) = (1-R) * Phi * alpha * exp(-alpha*x)
- Currently simplified; proper profile improves PDE and spectral response

## Medium Priority

### Trap Dynamics in Afterpulsing
- Multiple trap levels with distinct capture cross-sections and emission times
- Current model: single-trap exponential decay
- Extension: sum of exponentials for multi-trap capture/release
- Real InGaAs/InP SPADs have traps at heterojunction interfaces

### Avalanche Quenching Dynamics
- Spatial propagation of avalanche front across multiplication region
- Non-local feedback: carrier density at one point affects field at another
- Important for understanding spurious pulsing and crosstalk

### Multiplication Noise (Full Statistics)
- Dead-space multiplication theory (Hayat, Saleh)
- Beyond McIntyre's local model for thin multiplication regions
- Provides complete probability distribution of gain M
- Critical for receiver sensitivity calculations

## Low Priority

### 2D/3D Breakdown and Edge Effects
- Current model: 1D center-axis only
- Real SPADs have perimeter breakdown at guard ring edges
- Requires 2D Poisson + 2D transport
- Guard ring design optimization

### Hot-Carrier Luminescence
- Secondary photon emission during avalanche
- Can trigger afterpulsing in neighboring pixels (optical crosstalk)
- Relevant for SPAD arrays

### Temperature-Dependent Trap Parameters
- Capture cross-sections and emission rates vary with T
- Currently afterpulsing model uses fixed tau_c
- Needed for accurate DCR vs temperature modeling
