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
python inference_pipeline/refine_model_response.py \
  --split $SPLIT \
  --responses $INIT_RESPONSES \
  --refine_model $MODEL \
  --out_path $OUT_PATH \
  --refine_prompt "Your task is to generate text based on the provided task description. Additionally, you are given an example output from another model (the \"reference\"). Use the reference to identify elements that are well-executed and should be retained or adapted in your response, while also improving upon areas where it falls short or does not align with the task description. Focus on producing a result that best meets the requirements of the task and outperforms the reference. Do not provide any additional explanation, include only your answer to the task description.\n\nTask description: {instruction}\n\nReference: \"{ai_response}\"\n\nAnswer:"