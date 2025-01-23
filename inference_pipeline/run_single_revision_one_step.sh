#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <split> <init_responses> <model> <out_path>"
    exit 1
fi

# Assign input arguments to variables
SPLIT=$1
INIT_RESPONSES=$2
MODEL=$3
OUT_PATH=$4

# print the input arguments
echo "split: $SPLIT"
echo "init_responses: $INIT_RESPONSES"
echo "model: $MODEL"
echo "out_path: $OUT_PATH"

export HF_HOME=/cs/snapless/gabis/gililior/hf_cache
export PYTHONPATH=./
source /cs/snapless/gabis/gililior/virtual_envs/critic-routing/bin/activate

# Run the Python script with the provided arguments
python inference_pipeline/generate_initial_response.py \
  --split $SPLIT \
  --responses $INIT_RESPONSES \
  --refine_model $MODEL \
  --out_path $OUT_PATH