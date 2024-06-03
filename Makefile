# Makefile for ONSD Project

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
REQUIREMENTS = requirements.txt

NOTEBOOKS = \
    H2O_experimentation.ipynb \
    ONSD_autocrop.ipynb \
    ONSD_explore.ipynb \
    ONSD_exploreAP.ipynb \
    Optuna.ipynb \
    Other_models_ONSD.ipynb \
    Transfer_Learning.ipynb

# Default target
all: setup run

# Create virtual environment
$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

# Install dependencies
install: $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# Run the notebooks
run: install
	$(PYTHON) -m ipykernel install --user --name=$(VENV_DIR)
	for notebook in $(NOTEBOOKS); do \
	    jupyter nbconvert --to notebook --execute $$notebook; \
	done

# Clean up
clean:
	rm -rf $(VENV_DIR)
	find . -name "__pycache__" -exec rm -rf {} +

# Help
help:
	@echo "Makefile for ONSD Project"
	@echo ""
	@echo "Usage:"
	@echo "  make all       - Set up environment and run the notebooks"
	@echo "  make setup     - Set up the environment"
	@echo "  make install   - Install dependencies"
	@echo "  make run       - Run the notebooks"
	@echo "  make clean     - Clean up the environment"
	@echo "  make help      - Display this help message"

.PHONY: all setup install run clean help
