#!/bin/bash
#SBATCH --array=0-12
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:24g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=revisions
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

llama_models=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
gemma_models=("google/gemma-2-2b-it" "google/gemma-2-27b-it")

dataset="/cs/snapless/gabis/gililior/arena_data_final/constrained-lmsys-chat-1m"
split="train"
tasks_key="task"
out_dir="/cs/snapless/gabis/gililior/arena_data_v2/revisions"
init_response_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
len_llama=${#llama_models[@]}
len_gemma=${#gemma_models[@]}
total_combinations=$((len_llama * len_llama + len_gemma * len_gemma))

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0


for ((i=0; i<${#llama_models[@]}; i++)); do
  for ((j=0; j<${#llama_models[@]}; j++)); do
    if [ $i -eq $j ]; then
      continue
    fi
    generator="${llama_models[i]}"
    revision_model="${llama_models[j]}"
    generator_no_family=$(echo generator | sed 's/.*\///')
    revision_no_family=$(echo revision_model | sed 's/.*\///')
    path_to_init_response_generator="${init_response_dir}/${generator_no_family}-${split}-init-gen.json"
    out_path="${out_dir}/${revision_no_family}-revise-one-step-${generator_no_family}-${split}.json"
    if [ $index -eq $task_id ]; then
      echo "Running combination: generator=$generator, revision_model=$revision_model"
      ./inference_pipeline/run_single_revision_one_step.sh $split $path_to_init_response_generator $generator $out_path
      exit 0
    fi
    index=$((index + 1))
  done
done


for ((i=0; i<${#gemma_models[@]}; i++)); do
  for ((j=0; j<${#gemma_models[@]}; j++)); do
    if [ $i -eq $j ]; then
      continue
    fi
    generator="${gemma_models[i]}"
    revision_model="${gemma_models[j]}"
    generator_no_family=$(echo generator | sed 's/.*\///')
    revision_no_family=$(echo revision_model | sed 's/.*\///')
    path_to_init_response_generator="${init_response_dir}/${generator_no_family}-${split}-init-gen.json"
    out_path="${out_dir}/${revision_no_family}-revise-one-step-${generator_no_family}-${split}.json"
    if [ $index -eq $task_id ]; then
      echo "Running combination: generator=$generator, revision_model=$revision_model"
      ./inference_pipeline/run_single_revision_one_step.sh $split $path_to_init_response_generator $generator $out_path
      exit 0
    fi
    index=$((index + 1))
  done
done
