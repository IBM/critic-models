#!/bin/bash
#SBATCH --array=0-255%10  # Limit to 10 parallel jobs
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=hyperparam_search
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

# Define hyperparameter grid
learning_rates=(2e-5 3e-5)
batch_sizes=(2 3)
num_epochs=(3 5)
weight_decays=(0.01 0.1)
lora_rs=(8 16)
lora_alphas=(16 32)
lora_dropouts=(0.1 0.2)
seeds=(42 123)

config="utils/config_for_routing.json"
df="_output/data_for_critic_routing.csv"
model_name="google/gemma-2-2b-it"
results_file="_output/binary_classification_results_v2.csv"

# Calculate total number of combinations
total_combinations=$(( ${#learning_rates[@]} * ${#batch_sizes[@]} * ${#num_epochs[@]} * ${#weight_decays[@]} * ${#lora_rs[@]} * ${#lora_alphas[@]} * ${#lora_dropouts[@]} * ${#seeds[@]} ))

# Ensure results file exists with a header
if [ ! -f "$results_file" ]; then
  echo "model_name,learning_rate,batch_size,num_epochs,weight_decay,lora_r,lora_alpha,lora_dropout,seed,f1,accuracy" > "$results_file"
fi

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for epochs in "${num_epochs[@]}"; do
      for wd in "${weight_decays[@]}"; do
        for r in "${lora_rs[@]}"; do
          for alpha in "${lora_alphas[@]}"; do
            for dropout in "${lora_dropouts[@]}"; do
              for seed in "${seeds[@]}"; do
                if [ $index -eq $task_id ]; then
                  echo "Running combination: LR=$lr, BS=$bs, EPOCHS=$epochs, WD=$wd, R=$r, ALPHA=$alpha, DROPOUT=$dropout, SEED=$seed"
                  ./routing/run_training.sh $config $df $model_name $lr $bs $epochs $wd $r $alpha $dropout $seed $results_file
                  exit 0
                fi
                index=$((index + 1))
              done
            done
          done
        done
      done
    done
  done
done
