#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 12 ]; then
    echo "Usage: $0 <config> <df> <model_name> <output_dir> <learning_rate> <batch_size> <num_epochs> <weight_decay> <lora_r> <lora_alpha> <lora_dropout> <seed> <results_file>"
    exit 1
fi

# Assign input arguments to variables
CONFIG=$1
DF=$2
MODEL_NAME=$3
OUTPUT_DIR=$4
LEARNING_RATE=$5
BATCH_SIZE=$6
NUM_EPOCHS=$7
WEIGHT_DECAY=$8
LORA_R=$9
LORA_ALPHA=${10}
LORA_DROPOUT=${11}
SEED=${12}
RESULTS_FILE=${13}

# Run the Python script with the provided arguments
python routing/train_self_critic_prediction.py \
  --config $CONFIG \
  --df $DF \
  --model_name $MODEL_NAME \
  --output_dir $OUTPUT_DIR \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --weight_decay $WEIGHT_DECAY \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --seed $SEED \
  --results_file $RESULTS_FILE
