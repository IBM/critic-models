#!/bin/bash
#SBATCH --array=0-40%10  # Limit to 10 parallel jobs
#SBATCH --mem=80gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=multiclass_classification_ft
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

# Define hyperparameter grid
class_names=('self-critic' 'multi-class' 'prob-multi-class' 'regressor')
learning_rates=(1e-6 3e-6 1e-5 3e-5 1e-4)
#batch_sizes=(2 3)
batch_sizes=(2)
#num_epochs=(3 5)
num_epochs=(5)
weight_decays=(0.01 0.1)
#lora_rs=(8 16)
lora_rs=(8)
#lora_alphas=(16 32)
lora_alphas=(16)
#lora_dropouts=(0.1 0.2)
lora_dropouts=(0.1)
#seeds=(42 123)
seeds=(42)

config="utils/config_for_routing.json"
df="_output/data_for_critic_routing.csv"
model_name="mistralai/Ministral-8B-Instruct-2410"
results_file="_output/binary_classification_results_v2.csv"

# Calculate total number of combinations
total_combinations=$(( ${#learning_rates[@]} * ${#batch_sizes[@]} * ${#num_epochs[@]} * ${#weight_decays[@]} * ${#lora_rs[@]} * ${#lora_alphas[@]} * ${#lora_dropouts[@]} * ${#seeds[@]} ))


# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0
for cn in "${class_names[@]}"; do
  results_file="_output/${cn}_classification_results.csv"
  for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for epochs in "${num_epochs[@]}"; do
        for wd in "${weight_decays[@]}"; do
          for r in "${lora_rs[@]}"; do
            for alpha in "${lora_alphas[@]}"; do
              for dropout in "${lora_dropouts[@]}"; do
                for seed in "${seeds[@]}"; do
                  if [ $index -eq $task_id ]; then
                    echo "Running combination: class=$cn LR=$lr, BS=$bs, EPOCHS=$epochs, WD=$wd, R=$r, ALPHA=$alpha, DROPOUT=$dropout, SEED=$seed"
                    ./routing/run_training_v2.sh $cn $config $df $model_name $lr $bs $epochs $wd $r $alpha $dropout $seed $results_file
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
done
