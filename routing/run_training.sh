#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 12 ]; then
    echo "Usage: $0 <config> <df> <model_name> <learning_rate> <batch_size> <num_epochs> <weight_decay> <lora_r> <lora_alpha> <lora_dropout> <seed> <results_file>"
    exit 1
fi

# Assign input arguments to variables
CONFIG=$1
DF=$2
MODEL_NAME=$3
LEARNING_RATE=$4
BATCH_SIZE=$5
NUM_EPOCHS=$6
WEIGHT_DECAY=$7
LORA_R=$8
LORA_ALPHA=$9
LORA_DROPOUT=${10}
SEED=${11}
RESULTS_FILE=${12}

# print the input arguments
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
python routing/train_self_critic_prediction.py \
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
