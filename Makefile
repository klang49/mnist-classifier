# Define variables
PYTHON = python
PIP = pip3
BIN = bin
VENV_DIR = $(BIN)/venv
REQUIREMENTS = requirements.txt
SCRIPT = src/train.py  # Replace with the actual name of your script

# Default target
.PHONY: all
all: install

# Install mode: Create a virtual environment and install dependencies
.PHONY: install
install:
	@echo "Installing python3.10-venv package..."
	sudo apt update && sudo apt install -y python3.10-venv
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_DIR)/bin/activate && $(PIP) install --upgrade pip && $(PIP) install -r $(REQUIREMENTS)
	@echo "Installation complete."

# Train mode: Run the training script
.PHONY: train
train:
	@echo "Activating virtual environment and running training script..."
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(SCRIPT) --train
	@echo "Training complete."

# Analysis mode: Run the analysis script
.PHONY: analysis
analysis:
	@echo "Activating virtual environment and running analysis script..."
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(SCRIPT) --analysis
	@echo "Analysis complete."

# Remove mode: Remove the virtual environment
.PHONY: remove
remove:
	@echo "Removing virtual environment..."
	rm -rf $(BIN)/*
	@echo "Cleanup complete."
