# Makefile for RWKV-based Twitch Chat Project

# Default target
.DEFAULT_GOAL := run-model

# Python interpreter
PYTHON := python3

# Paths
DATA_DIR := data
MODEL_DIR := models
CHECKPOINT_DIR := $(MODEL_DIR)/checkpoints

# Target for installing dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Target to train the RWKV model
train:
	$(PYTHON) train_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR) --checkpoint_dir $(CHECKPOINT_DIR)

# Target to test the RWKV model
test:
	$(PYTHON) test_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR)

# Run the model for generating chat
run-model:
	$(PYTHON) run_model.py --model_dir $(MODEL_DIR)
