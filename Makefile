# Makefile for RWKV-based Twitch Chat Project

# Default target
.DEFAULT_GOAL := run-model

# Python interpreter
PYTHON := python3

# Paths
DATA_DIR := data
MODEL_DIR := models
CHECKPOINT_DIR := $(MODEL_DIR)/checkpoints

# Target for installing Python dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Target for installing dependencies with Homebrew
install-brew:
	brew install blackhole-2ch
	python3 device_select.py

# Target for uninstalling dependencies with Homebrew
uninstall-brew:
	brew uninstall blackhole-2ch

# Placeholder for the main Python script
run-main:
	@echo "The main Python script does not exist yet. Please implement it."

# Run the transcription script
run-transcription:
	$(PYTHON) live_transcription.py

# Combine all JSON files into train.json only
combine-training-data-only:
	$(PYTHON) merge_train_json.py

# Combine JSON files into train.json and test.json with a test split
combine-training-data-test-split:
	@if test $(words $(MAKECMDGOALS)) -eq 1; then \
		$(PYTHON) merge_json.py --test_split=0.2; \
	else \
		SPLIT_VALUE=$(word 2,$(MAKECMDGOALS)); \
		if ! echo "$$SPLIT_VALUE" | grep -Eq '^(0(\.[0-9]+)?|1(\.0)?)$$' ; then \
			echo "Error: Split value must be between 0 and 1.0"; \
			exit 1; \
		fi; \
		$(PYTHON) merge_json.py --test_split=$$SPLIT_VALUE; \
	fi

# Target to train the RWKV model
train:
	$(PYTHON) train_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR) --checkpoint_dir $(CHECKPOINT_DIR)

# Target to test the RWKV model
test:
	$(PYTHON) test_model.py --data_dir $(DATA_DIR) --model_dir $(MODEL_DIR)

# Run the model for generating chat
run-model:
	$(PYTHON) run_model.py --model_dir $(MODEL_DIR)

# Record audio clips with default (built-in microphone)
record-audio:
	$(PYTHON) async_segmented_transcription.py record default

# Record audio clips with virtual (BlackHole microphone)
virtual-record-audio:
	$(PYTHON) async_segmented_transcription.py record virtual

# Process audio clips into transcriptions
process-audio:
	$(PYTHON) async_segmented_transcription.py process

# Convert transcriptions to JSON
transcriptions-to-json:
	$(PYTHON) process_transcriptions_json.py --input_dir text_transcriptions --output_dir transcribed_inputs

# Splits the files in json_splitter/ and splits them into chunks designated in json_splitter.py
json-split:
	$(PYTHON) json_splitter.py

