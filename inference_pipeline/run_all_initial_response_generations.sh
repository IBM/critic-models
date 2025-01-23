#!/bin/bash
#SBATCH --array=0-3
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=initial_generations
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

model_names=("meta-llama/Llama-3.1-70B-Instruct" "google/gemma-2-27b-it" "microsoft/Phi-3.5-mini-instruct" "microsoft/Phi-3.5-MoE-instruct")

dataset="/cs/snapless/gabis/gililior/arena_data_final/constrained-lmsys-chat-1m"
split="train"
tasks_key="task"
out_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
total_combinations=${#model_names[@]}

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

for model in "${model_names[@]}"; do
  # remove the model directory (until separator) from the model name
  model_name_no_family=$(echo $model | sed 's/.*\///')
  out_path="${out_dir}/${model_name_no_family}-${split}-init-gen.json"
  if [ $index -eq $task_id ]; then
    echo "Running combination: MODEL=$model"
    ./inference_pipeline/run_single_initial_generation.sh $dataset $split $tasks_key $model $out_path
    exit 0
  fi
  index=$((index + 1))
done