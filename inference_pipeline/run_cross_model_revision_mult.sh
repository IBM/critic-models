#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset> <model> <out_path> <num_iterations> <starting_index>"
    exit 1
fi

# Assign input arguments to variables
DATASET=$1
MODEL=$2
OUT_PATH=$3
NUM_ITERATIONS=$4
STARTING_INDEX=$5

# print the input arguments
echo "dataset: $DATASET"
echo "model: $MODEL"
echo "out_path: $OUT_PATH"
echo "num_iterations: $NUM_ITERATIONS"
echo "starting_index: $STARTING_INDEX"


export HF_HOME=/cs/snapless/gabis/gililior/hf_cache
export PYTHONPATH=./
source /cs/snapless/gabis/gililior/virtual_envs/critic-routing/bin/activate

python inference_pipeline/generate_multiple_other_model_revisions_in_context.py \
  --dataset $DATASET \
  --model $MODEL \
  --out_path $OUT_PATH \
  --num_iterations $NUM_ITERATIONS \
  --starting_index $STARTING_INDEX