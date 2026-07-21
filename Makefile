PYTHON = .venv/bin/python
# PYTHONPATH points to parent so that 'import spad_simulator' resolves
# the root __init__.py at spad_simulator/__init__.py
ROOT = ..

.PHONY: run run-iv run-ui run-quick clean clean-plots clean-pyc check typecheck help deploy-plots

run:   ## Run the full SPAD simulation
	PYTHONWARNINGS=ignore PYTHONPATH=$(ROOT) $(PYTHON) -m spad_simulator
	$(MAKE) deploy-plots

run-ui:  ## Launch the SPAD simulation GUI
	PYTHONWARNINGS=ignore PYTHONPATH=$(ROOT) $(PYTHON) -m spad_simulator --ui

run-quick:  ## Run with minimal output (WARNING level only)
	PYTHONWARNINGS=ignore PYTHONPATH=$(ROOT) $(PYTHON) -c "import logging; from spad_simulator.src.utils._logging import set_log_level; set_log_level(logging.WARNING); from spad_simulator.src.main import main; main()"

run-iv:  ## Run IV characteristic plot only (quick iteration)
	PYTHONWARNINGS=ignore PYTHONPATH=$(ROOT) $(PYTHON) -m spad_simulator --iv-only

notebooks:  ## Set up venv kernel for learning notebooks
	$(PYTHON) -m ipykernel install --user --name="spad-sim" --display-name="SPAD Sim" 2>/dev/null || true

check:   ## Check for import / syntax errors without running
	PYTHONPATH=$(ROOT) $(PYTHON) -c "import spad_simulator; print('Package OK')"

clean-plots:  ## Remove all generated plots
	rm -rf plots/ docs/plots/

clean-pyc:  ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

clean: clean-plots clean-pyc  ## Remove plots and cache files

typecheck:  ## Run mypy type checking
	PYTHONPATH=$(ROOT) $(PYTHON) -m mypy spad_simulator/ --ignore-missing-imports 2>/dev/null || echo "mypy not installed; skipping"

deploy-plots:  ## Copy plots + data to docs/ for GitHub Pages
	mkdir -p docs/plots/spad docs/data
	cp -n plots/spad/*.png docs/plots/spad/ 2>/dev/null || true
	cp -n data/*.xml docs/data/ 2>/dev/null || true

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
