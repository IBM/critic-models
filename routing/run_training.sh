#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 11 ]; then
    echo "Usage: $0 <df> <model_name> <output_dir> <learning_rate> <batch_size> <num_epochs> <weight_decay> <lora_r> <lora_alpha> <lora_dropout> <seed>"
    exit 1
fi

# Assign input arguments to variables
DF=$1
MODEL_NAME=$2
OUTPUT_DIR=$3
LEARNING_RATE=$4
BATCH_SIZE=$5
NUM_EPOCHS=$6
WEIGHT_DECAY=$7
LORA_R=$8
LORA_ALPHA=$9
LORA_DROPOUT=${10}
SEED=${11}

# Run the Python script with the provided arguments
python train_model.py \
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
  --seed $SEED
