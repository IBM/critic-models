#!/bin/bash
#SBATCH --array=0-9
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=initial_generations_with_temp
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

model="meta-llama/Llama-3.1-8B-Instruct"

dataset="gililior/wild-if-eval"
tasks_key="task"
out_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations_with_temp"

split="test"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
total_combinations=10

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

for ((i=0; i<$total_combinations; i++)); do
  # remove the model directory (until separator) from the model name
  model_name_no_family=$(echo $model | sed 's/.*\///')
  out_path="${out_dir}/${model_name_no_family}-${split}-init-gen-${i}.json"
  if [ $index -eq $task_id ]; then
    echo "Running combination: MODEL=$model, i=$i"
    ./inference_pipeline/run_single_initial_generation.sh $dataset $split $tasks_key $model $out_path
    exit 0
  fi
  index=$((index + 1))
done