#!/bin/bash
#
# AUTHOR:
# @chizo4 (Filip J. Cierkosz)
#
# INFO:
# The script runs the full-experiment pipeline for movie classification.
#
# NOTE:
# The currently implemented code supports any movie data collection
# provided it follows the same file organization as the "imdb" directory
# (e.g., contains train-*.csv data in CSV).
#
# USAGE:
# bash script/run-pipeline.sh [data_path] [val_csv] [test_csv] [direct_json] [write_json]
#
# EXAMPLE (for the task of "IMDB"):
# bash script/run-pipeline.sh imdb validation_hidden.csv test_hidden.csv directing.json writing.json
#
# OPTIONS:
# [data_path]   -> Base path to access the directory with movie data.
# [val_csv]     -> Name of the validation CSV file. Just filename, since we assume it is in data_path.
# [test_csv]    -> Name of the test CSV file. Just filename, since we assume it is in data_path.
# [direct_json] -> Name of the directing JSON file, assuming it is in data_path.
# [write_json]  -> Name of the writing JSON file, assuming it is in data_path.

########################### CONFIGURATION & SETUP ###########################

# STEP 0: Check if the correct number of arguments is provided.
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 [data_path] [val_csv] [test_csv] [direct_json] [write_json]"
    exit 1
fi

# STEP 1: Fetch CLI args.
DATA_PATH=$1
VAL_CSV=$2
TEST_CSV=$3
DIRECT_JSON=$4
WRITE_JSON=$5

# STEP 2: Setup for input/output resources.
echo "START: Task initialization - verifying configuration..."; echo
# (A) Input directory (data).
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Directory '$DATA_PATH' not found. EXIT!"; echo
    exit 1
fi
# (B) Verify the existence of passed files.
for file in "$DATA_PATH/$VAL_CSV" "$DATA_PATH/$TEST_CSV" "$DATA_PATH/$DIRECT_JSON" "$DATA_PATH/$WRITE_JSON"; do
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

echo "PIPELINE: Initial setup verified. Proceeding with pipeline..."; echo

########################### RUN PIPELINE ###########################

# STEP 3: Activate targe conda environment.
ENV_NAME="movie_classifier_env"
echo "Activating '$ENV_NAME' environment..."; echo
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# STEP 4: Run the pipeline.
python movie_pipeline/classifier_pipeline.py \
    --data "$DATA_PATH" \
    --val "$VAL_CSV" \
    --test "$TEST_CSV" \
    --directing "$DIRECT_JSON" \
    --writing "$WRITE_JSON"
