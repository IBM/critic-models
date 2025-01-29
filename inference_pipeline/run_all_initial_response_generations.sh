#!/bin/bash
#SBATCH --array=0-2
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=initial_generations
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

#model_names=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-2b-it" "google/gemma-2-9b-it")
#model_names=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
model_names=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct")  # "Qwen/Qwen2.5-7B-Instruct")


#dataset="/cs/snapless/gabis/gililior/arena_data_final/constrained-lmsys-chat-1m"
#tasks_key="task"
#out_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations"
dataset="google/IFEval"
tasks_key="prompt"
out_dir="/cs/snapless/gabis/gililior/if-eval-generations/initial"
split="train"
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