#!/bin/bash

models=("mistral-large" "deepseek-v3")
models=("llama3.3-70b" "qwen2.5-72b")

init_response_dir="/path/to/repo/wild-if-eval-code/model_predictions"
out_dir="/path/to/out/dir"
mkdir -p $out_dir


for ((i=0; i<${#models[@]}; i++)); do
  for ((j=0; j<${#models[@]}; j++)); do
    generator="${models[i]}"
    revision_model="${models[j]}"
    path_to_init_response_generator="${init_response_dir}/${generator}-0shot-wild-if-eval.json"
    echo "Running combination: generator=$generator, revision_model=$revision_model"
    python inference_pipeline/big_models_revisions_rits.py \
      --data_path $path_to_init_response_generator \
      --model $revision_model \
      --out_dir out_dir \
      --tasks_batch_size 200
  done
done