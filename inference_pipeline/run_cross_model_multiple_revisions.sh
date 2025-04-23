#!/bin/bash
#SBATCH --array=0-23
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=cross_model_multiple_revisions
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

models=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-3-4b-it", "google/gemma-3-12b-it")

init_response_dir="/cs/snapless/gabis/gililior/if-eval-generations/multiple_revisions/"
out_dir="/cs/snapless/gabis/gililior/if-eval-generations/cross-model-revisions"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
len_models=${#models[@]}
total_combinations=$((2*(len_models * (len_models-1))))

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

num_iterations=4

for starting_index in "0 1"; do
  for ((i=0; i<${#models[@]}; i++)); do
    for ((j=0; j<${#models[@]}; j++)); do
      if [ $i -eq $j ]; then
        continue
      fi
      generator="${models[i]}"
      revision_model="${models[j]}"
      generator_no_family=$(echo $generator | sed 's/.*\///')
      revision_no_family=$(echo $revision_model | sed 's/.*\///')
      path_to_init_response_generator="${init_response_dir}/${generator_no_family}.json"
      out_path="${out_dir}/${revision_no_family}-revise-multiple-${generator_no_family}.json"
      if [ $index -eq $task_id ]; then
        if [ ! -f $out_path ]; then
          echo "Running combination: generator=$generator, revision_model=$revision_model"
          ./inference_pipeline/run_single_revision_one_step.sh $path_to_init_response_generator $revision_model $out_path $num_iterations $starting_index
          exit 0
        fi
      fi
      index=$((index + 1))
    done
  done
done