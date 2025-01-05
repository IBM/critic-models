#!/bin/bash

run_name="${3}_model_${4}_lr_${5}_bs_${6}_epochs_${7}_wd_${8}_r_${9}_alpha_${10}_dropout_${11}_seed_${12}"

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
echo "run_name: $run_name"

#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=1-12
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=${run_name}/error_log_job%A.txt
#SBATCH --output=${run_name}/output_log_job%A.txt
#SBATCH --job-name=${run_name}
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

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
