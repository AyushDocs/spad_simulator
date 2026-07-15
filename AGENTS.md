# AGENTS.md — SPAD Simulator

## Quick Reference

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Run the full simulation (CLI)
make run

# Run the simulation GUI
make run-ui

# Run tests
pytest

# Run a single test file
pytest tests/test_core.py

# Type checking (many error codes disabled)
make typecheck

# Lint (ruff, line-length 100)
ruff check src/ tests/

# Quick import check (no actual run)
make check
```

## Package Structure

The repo root contains a thin wrapper package (`__init__.py`, `__main__.py`) that imports from `src/`. All real code lives under `src/`.

```
src/
  core/           Grid, layers, materials, doping, device
  poisson/        Nonlinear Poisson solver, depletion width
  avalanche/      Impact ionization, trigger, breakdown, dark current, PDE
  transport/      Drift-diffusion, Monte Carlo, timing jitter
  self_consistent/ PIC loop, particle-mesh, circuit quenching
  optimization/   PSO optimizer
  studies/        High-level simulation workflows (fields, dark current, PDE, etc.)
  utils/          Exceptions, logging, XML loaders, plotters, ingestion, artifacts
  main.py         Orchestration entrypoint
  ui.py           GUI mode
data/             XML device specs, material params, absorption data
tests/            pytest tests (mirrors src/ module structure)
```

## Key Conventions

- **Python ≥3.12** required
- **ruff** for linting: line-length 100, ignores F841/F401/F821/E741/E501/E402/I001
- **mypy** with `ignore_missing_imports = true`; many error codes disabled (no-any-return, arg-type, index, name-defined, call-arg, operator)
- All spatial integrations use **trapezoidal rule** (`np.trapezoid`), not midpoint/Riemann
- Data files are XML; loaded via `DataIngestionService` from `data/`
- Plots saved to `plots/spad/`; the dir is gitignored

## Run Behavior

- `make run` sets `PYTHONPATH=..` so `import spad_simulator` resolves the root `__init__.py`
- `python -m src` from repo root also works (README method)
- `--ui` flag switches from CLI to GUI mode
- Output: 15 diagnostic plots + `sim_results.xml` in `plots/spad/`

## Testing

- Tests import directly from `src.*` (e.g., `from src.core.material import Material`)
- Shared fixtures in `tests/conftest.py` (e.g., `absorption_data`, `inp_material`)
- No external services or databases required; tests are pure computation

## Gotchas

- `make run` needs `ROOT=..` in Makefile; running `python -m spad_simulator` without PYTHONPATH will fail
- `plots/` is gitignored — do not commit generated plots
- mypy baseline is lenient; many type errors are intentionally suppressed
