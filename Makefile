# Makefile for Python Project

# Variables
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Default target
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv       - Create a virtual environment."
	@echo "  make install    - Install dependencies."
	@echo "  make build      - Build the project."
	@echo "  make run        - Run the application."
	@echo "  make test       - Run tests."
	@echo "  make lint       - Run linters."
	@echo "  make format     - Format code with Black."
	@echo "  make clean      - Clean up build artifacts."
	@echo "  make activate   - Activate the virtual environment."

# Create a virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Install dependencies
.PHONY: install
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Build the project (e.g., compile extensions, prepare distributions)
.PHONY: build
build: install
	@echo "Building the project..."
	# Add build commands here if needed

# Run the application
.PHONY: run
run: install
	$(PYTHON) src/qa-system-rag/main.py $(ARGS)

# Run tests
.PHONY: test
test: install
	$(PYTHON) -m unittest discover -s tests

# Run linters
.PHONY: lint
lint: install
	$(PIP) install flake8
	flake8 src/ tests/

# Format code
.PHONY: format
format: install
	$(PIP) install black
	black src/ tests/

# Clean up build artifacts
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf dist build

# Instructions to activate the virtual environment
.PHONY: activate
activate:
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV_DIR)/bin/activate"
