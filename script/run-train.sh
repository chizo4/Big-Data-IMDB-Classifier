#!/bin/bash
#
# AUTHOR:
# @chizo4 (Filip J. Cierkosz)
#
# INFO:
# The script runs the TRAINING via pipeline for movie classification.
#
# NOTE:
# The currently implemented code supports any movie data collection
# provided it follows the same file organization as the "imdb" directory
# (e.g., contains train-*.csv data in CSV). Same for ollama models - supports
# any choices as long as they are pull via ollama. We recommend gemma3 though.
#
# USAGE:
# bash script/run-train.sh [data_path] [direct_json] [write_json] [model]
#
# EXAMPLE (for the task of "IMDB"):
# bash script/run-train.sh imdb directing.json writing.json gemma3:1b
#
# OPTIONS:
# [data_path]   -> Base path to access the directory with movie data.
# [direct_json] -> Name of the directing JSON file, assuming it is in data_path.
# [write_json]  -> Name of the writing JSON file, assuming it is in data_path.
# [model]       -> Name of the LLM provided in ollama API.

########################### CONFIGURATION & SETUP ###########################

# STEP 0: Check if the correct number of arguments is provided.
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 [data_path] [direct_json] [write_json] [model]"
    exit 1
fi

# STEP 1: Fetch CLI args and pre-process.
DATA_PATH=$1
DIRECT_JSON="$DATA_PATH/$2"
WRITE_JSON="$DATA_PATH/$3"
MODEL=$4

# STEP 2: Setup for input/output resources.
echo "START: TRAIN task initialization - verifying configuration..."; echo
# (A) Input directory (data).
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Directory '$DATA_PATH' not found. EXIT!"; echo
    exit 1
fi
# (B) Verify the existence of passed files.
for file in "$DIRECT_JSON" "$WRITE_JSON"; do
    if [ ! -f "$file" ]; then
        echo "$file"
        echo "ERROR: File '$file' not found. EXIT!"; echo
        exit 1
    fi
done
# (C) Output directory (results).
RESULTS_DIR="results/$DATA_PATH"
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "INFO: Created RESULTS directory: $RESULTS_DIR"; echo
fi
# (D) Model directory.
MODEL_DIR="model/$DATA_PATH"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    echo "INFO: Created MODEL directory: $MODEL_DIR"; echo
fi

echo "TRAINING: Initial setup verified. Proceeding with pipeline..."; echo

########################### RUN PIPELINE ###########################

# STEP 3: Activate target conda environment.
ENV_NAME="movie_classifier_env"
echo "Activating '$ENV_NAME' environment..."; echo
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# STEP 4: Run the TRAIN via pipeline.
echo "***TRAINING: TASK INITIALIZATION:***"; echo
python movie_pipeline/train.py \
    --data-path "$DATA_PATH" \
    --directing-json "$DIRECT_JSON" \
    --writing-json "$WRITE_JSON" \
    --model-path "$MODEL_DIR" \
    --results-path "$RESULTS_DIR" \
    --model "$MODEL"
