#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 13 ]; then
    echo "Usage: $0 <class_name> <config> <df> <model_name> <learning_rate> <batch_size> <num_epochs> <weight_decay> <lora_r> <lora_alpha> <lora_dropout> <seed> <results_file>"
    exit 1
fi

# Assign input arguments to variables
CLASS_NAME=$1
CONFIG=$2
DF=$3
MODEL_NAME=$4
LEARNING_RATE=$5
BATCH_SIZE=$6
NUM_EPOCHS=$7
WEIGHT_DECAY=$8
LORA_R=$9
LORA_ALPHA=${10}
LORA_DROPOUT=${11}
SEED=${12}
RESULTS_FILE=${13}

# print the input arguments
echo "class_name: $CLASS_NAME"
echo "config: $CONFIG"
echo "df: $DF"
echo "model_name: $MODEL_NAME"
echo "learning_rate: $LEARNING_RATE"
echo "batch_size: $BATCH_SIZE"
echo "num_epochs: $NUM_EPOCHS"
echo "weight_decay: $WEIGHT_DECAY"
echo "lora_r: $LORA_R"
echo "lora_alpha: $LORA_ALPHA"
echo "lora_dropout: $LORA_DROPOUT"
echo "seed: $SEED"
echo "results_file: $RESULTS_FILE"

export HF_HOME=/cs/snapless/gabis/gililior/hf_cache
export PYTHONPATH=./
source /cs/snapless/gabis/gililior/virtual_envs/critic-routing/bin/activate

# Run the Python script with the provided arguments
python routing/run_training.py \
  --class_name $CLASS_NAME \
  --config $CONFIG \
  --df $DF \
  --model_name $MODEL_NAME \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --weight_decay $WEIGHT_DECAY \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --seed $SEED \
  --results_file $RESULTS_FILE
