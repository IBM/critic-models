#!/bin/bash
#SBATCH --array=0-4
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=revisions
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL


split="train"
out_dir="/cs/snapless/gabis/gililior/arena_data_v2/revisions"
init_response_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations"
#init_response_dir="/cs/snapless/gabis/gililior/if-eval-generations/initial"
#out_dir="/cs/snapless/gabis/gililior/if-eval-generations/revisions"
mkdir -p $out_dir
mkdir -p slurm_logs

pairs=("meta-llama/Llama-3.2-1B-Instruct,google/gemma-2-2b-it" "meta-llama/Llama-3.2-3B-Instruct,google/gemma-2-9b-it" "google/gemma-2-2b-it,meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct,google/gemma-2-9b-it" "google/gemma-2-9b-it,meta-llama/Llama-3.1-8B-Instruct")

# Calculate total number of combinations
total_combinations=${#pairs[@]}


# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0

for pair in "${pairs[@]}"; do
  IFS=',' read -r generator revision_model <<< "$pair"
  echo $generator
  echo $revision_model
  generator_no_family=$(echo $generator | sed 's/.*\///')
  revision_no_family=$(echo $revision_model | sed 's/.*\///')
  path_to_init_response_generator="${init_response_dir}/${generator_no_family}-${split}-init-gen.json"
  out_path="${out_dir}/${revision_no_family}-revise-one-step-${generator_no_family}-${split}.json"
  if [ ! -f $out_path ]; then
    echo "Running combination: generator=$generator, revision_model=$revision_model"
    ./inference_pipeline/run_single_revision_one_step.sh $path_to_init_response_generator $revision_model $out_path
  fi
done

