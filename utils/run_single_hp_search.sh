#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <lr> <batch_size> <epochs>"
    exit 1
fi

# Assign input arguments to variables
LR=$1
BS=$2
EPOCHS=$3

# print the input arguments
echo "lr: $LR"
echo "batch_size: $BS"
echo "epochs: $EPOCHS"

export HF_HOME=/cs/snapless/gabis/gililior/hf_cache
export PYTHONPATH=./
source /cs/snapless/gabis/gililior/virtual_envs/critic-routing/bin/activate

# Run the Python script with the provided arguments
python utils/finetune_modernbert.py \
  --lr $LR \
  --batch_size $BS \
  --epochs $EPOCHS