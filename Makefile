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
	@if test $(words $(MAKECMDGOALS)) -eq 1; then \
		$(PYTHON) train_model.py \
		--data_dir $(DATA_DIR) \
		--model_dir $(MODEL_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--max_seq_length 769; \
	else \
		MODEL_TYPE=$(word 2,$(MAKECMDGOALS)); \
		EPOCHS=$(word 3,$(MAKECMDGOALS)); \
		BATCH_SIZE=$(word 4,$(MAKECMDGOALS)); \
		VOCAB_SIZE=$(word 5,$(MAKECMDGOALS)); \
		MAX_SEQ_LENGTH=$(word 6,$(MAKECMDGOALS)); \
		LOAD_MODEL=$(word 7,$(MAKECMDGOALS)); \
		if [ -z "$$MODEL_TYPE" ]; then MODEL_TYPE="RWKV-4-World"; fi; \
		if [ -z "$$EPOCHS" ]; then EPOCHS="5"; fi; \
		if [ -z "$$BATCH_SIZE" ]; then BATCH_SIZE="16"; fi; \
		if [ -z "$$VOCAB_SIZE" ]; then VOCAB_SIZE="50257"; fi; \
		if [ -z "$$MAX_SEQ_LENGTH" ]; then MAX_SEQ_LENGTH="256"; fi; \
		if [ -z "$$LOAD_MODEL" ]; then LOAD_MODEL=""; fi; \
		if ! echo "$$EPOCHS" | grep -Eq '^[0-9]+$$'; then \
      		echo "Error: EPOCHS must be an integer value"; \
      		exit 1; \
		fi; \
		if ! echo "$$BATCH_SIZE" | grep -Eq '^[0-9]+$$'; then \
      		echo "Error: BATCH_SIZE must be an integer value"; \
      		exit 1; \
		fi; \
		if ! echo "$$VOCAB_SIZE" | grep -Eq '^[0-9]+$$'; then \
			echo "Error: VOCAB_SIZE must be an integer value"; \
			exit 1; \
		fi; \
		if ! echo "$$MAX_SEQ_LENGTH" | grep -Eq '^[0-9]+$$'; then \
			echo "Error: MAX_SEQ_LENGTH must be an integer value"; \
			exit 1; \
		fi; \
		$(PYTHON) train_model.py \
		--data_dir $(DATA_DIR) \
		--model_dir $(MODEL_DIR) \
		--checkpoint_dir $(CHECKPOINT_DIR) \
		--model_type $$MODEL_TYPE \
		--epochs $$EPOCHS \
		--batch_size $$BATCH_SIZE \
		--vocab_size $$VOCAB_SIZE \
		--max_seq_length $$MAX_SEQ_LENGTH \
		--load_model $$LOAD_MODEL; \
	fi

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

# Test out TextCleaner.py
text-clean:
	$(PYTHON) TextCleaner.py


