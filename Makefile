# Makefile for transcription project

# Default target
.DEFAULT_GOAL := run-main

# Python interpreter
PYTHON := python3

# Target for installing dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Run the transcription script
run-transcription:
	$(PYTHON) live_transcription.py

# Placeholder for the main Python script
run-main:
	@echo "The main Python script does not exist yet. Please implement it."
