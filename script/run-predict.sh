#!/bin/bash
#
# AUTHOR:
# @chizo4 (Filip J. Cierkosz)
#
# INFO:
# The script runs the prediction for movie classification.
#
# NOTE:
# The currently implemented code supports any movie data collection
# provided it follows the same file organization as the "imdb" directory
# (e.g., contains train-*.csv data in CSV). Same for ollama models - supports
# any choices as long as they are pull via ollama. We recommend gemma3 though.
#
# USAGE:
# bash script/run-predict.sh [data_path] [direct_json] [write_json] [test_csv] [model]
#
# EXAMPLE (for the task of "IMDB" on TEST data):
# bash script/run-predict.sh imdb directing.json writing.json test_hidden.csv gemma3:4b
#
# OPTIONS:
# [data_path]   -> Base path to access the directory with movie data.
# [direct_json] -> Name of the directing JSON file, assuming it is in data_path.
# [write_json]  -> Name of the writing JSON file, assuming it is in data_path.
# [test_csv]    -> Name of the validation CSV file. Just filename, since we assume it is in data_path.
# [model]       -> Name of the LLM provided in ollama API.

########################### CONFIGURATION & SETUP ###########################

# STEP 0: Check if the correct number of arguments is provided.
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 [data_path] [direct_json] [write_json] [test_csv] [model]"
    exit 1
fi

# STEP 1: Fetch CLI args and pre-process.
DATA_PATH=$1
DIRECT_JSON="$DATA_PATH/$2"
WRITE_JSON="$DATA_PATH/$3"
TEST_CSV="$DATA_PATH/$4"
MODEL=$5

# STEP 2: Setup for input/output resources.
echo "START: PREDICTION task initialization - verifying configuration..."; echo
# (A) Input directory (data).
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Directory '$DATA_PATH' not found. EXIT!"; echo
    exit 1
fi
# (B) Verify the existence of passed files.
for file in "$DIRECT_JSON" "$WRITE_JSON" "$TEST_CSV"; do
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

echo "PREDICTION: Initial setup verified. Proceeding with pipeline..."; echo

########################### RUN PIPELINE ###########################

# STEP 3: Activate targe conda environment.
ENV_NAME="movie_classifier_env"
echo "Activating '$ENV_NAME' environment..."; echo
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# STEP 4: Run PREDICT via pipeline.
python movie_pipeline/predict.py \
    --data-path "$DATA_PATH" \
    --directing-json "$DIRECT_JSON" \
    --writing-json "$WRITE_JSON" \
    --model-path "$MODEL_DIR" \
    --results-path "$RESULTS_DIR" \
    --test-csv "$TEST_CSV" \
    --model "$MODEL"
