#!/bin/bash
#SBATCH --array=0-9
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=revisions-bigger-models
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

generator="meta-llama/Llama-3.1-8B-Instruct"
revision_model="google/gemma-2-9b-it"

init_response_dir="/cs/snapless/gabis/gililior/wild-if-eval-code/model_predictions"
out_dir="/cs/snapless/gabis/gililior/wild-if-eval-revisions-with-temp"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
total_combinations=10

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0


for ((i=0; i<${total_combinations}; i++)); do
  generator_no_family=$(echo $generator | sed 's/.*\///' | sed 's/-Instruct//' | sed 's/-it//')
  revision_no_family=$(echo $revision_model | sed 's/.*\///' | sed 's/-Instruct//' | sed 's/-it//')
  path_to_init_response_generator="${init_response_dir}/${generator_no_family}-0shot-wild-if-eval.json"
  out_path="${out_dir}/${revision_no_family}-revise-one-step-${generator_no_family}-${i}.json"
  if [ $index -eq $task_id ]; then
    if [ ! -f $out_path ]; then
      echo "Running combination: generator=$generator, revision_model=$revision_model, i=$i"
      ./inference_pipeline/run_single_revision_one_step.sh $path_to_init_response_generator $revision_model $out_path
      exit 0
    fi
  fi
  index=$((index + 1))
done