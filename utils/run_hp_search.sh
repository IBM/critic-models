#!/bin/bash
#SBATCH --array=0-59
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=hp_search
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

learning_rates=(5e-5 1e-5 1e-6 1e-4 5e-3)
batch_sizes=(8 16 32 64)
epochs=(3 5 10)

num_learning_rates=${#learning_rates[@]}
num_batch_sizes=${#batch_sizes[@]}
num_epochs=${#epochs[@]}

mkdir -p slurm_logs

# Calculate total number of combinations
total_combinations=$((num_learning_rates * num_batch_sizes * num_epochs))

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for ep in "${epochs[@]}"; do
      if [ $index -eq $task_id ]; then
        echo "Running combination: LR=$lr, BS=$bs, EPOCHS=$ep"
        ./utils/run_single_hp_search.sh $lr $bs $ep
        exit 0
      fi
      index=$((index + 1))
    done
  done
done