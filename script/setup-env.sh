#!/bin/bash
#
# AUTHOR:
# @chizo4 (Filip J. Cierkosz)
#
# INFO:
# The script sets up the environment for the full-experiment pipeline for
# the movie classification project.

########################### CONFIGURATION & SETUP ###########################

ENV_NAME="movie_classifier_env"

# STEP 1: Create conda environment and activate it.
conda create --name $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# STEP 2: Install required dependencies.
pip install -r requirements.txt

echo; echo "INFO: Environment '$ENV_NAME' setup complete."; echo
