#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset> <split> <tasks_key> <model> <out_path>"
    exit 1
fi

# Assign input arguments to variables
DATASET=$1
SPLIT=$2
TASKS_KEY=$3
MODEL=$4
OUT_PATH=$5

# print the input arguments
echo "dataset: $DATASET"
echo "split: $SPLIT"
echo "tasks_key: $TASKS_KEY"
echo "model: $MODEL"
echo "out_path: $OUT_PATH"

export HF_HOME=/cs/snapless/gabis/gililior/hf_cache
export PYTHONPATH=./
source /cs/snapless/gabis/gililior/virtual_envs/critic-routing/bin/activate

# Run the Python script with the provided arguments
python inference_pipeline/generate_initial_response.py \
  --dataset $DATASET \
  --split $SPLIT \
  --tasks_key $TASKS_KEY \
  --model $MODEL \
  --out_path $OUT_PATH