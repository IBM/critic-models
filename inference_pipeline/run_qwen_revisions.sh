#!/bin/bash
#SBATCH --array=0-5
#SBATCH --mem=40gb
#SBATCH -c2
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1,vmem:40g
#SBATCH --error=slurm_logs/slurm_%A_%a.err
#SBATCH --output=slurm_logs/slurm_%A_%a.out
#SBATCH --job-name=revisions
#SBATCH --mail-user=gili.lior@mail.huji.ac.il
#SBATCH --mail-type=ALL

model_names=("Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct")


split="train"
#out_dir="/cs/snapless/gabis/gililior/arena_data_v2/revisions"
#init_response_dir="/cs/snapless/gabis/gililior/arena_data_v2/initial_generations"
init_response_dir="/cs/snapless/gabis/gililior/if-eval-generations/initial"
out_dir="/cs/snapless/gabis/gililior/if-eval-generations/revisions"
mkdir -p $out_dir
mkdir -p slurm_logs

# Calculate total number of combinations
total_combinations=${#model_names[@]}

# Calculate hyperparameter combination for this task
task_id=${SLURM_ARRAY_TASK_ID}
index=0


for ((i=0; i<${#model_names[@]}; i++)); do
  for ((j=i; j<${#model_names[@]}; j++)); do
    generator="${model_names[i]}"
    revision_model="${model_names[j]}"
    generator_no_family=$(echo $generator | sed 's/.*\///')
    revision_no_family=$(echo $revision_model | sed 's/.*\///')
    path_to_init_response_generator="${init_response_dir}/${generator_no_family}-${split}-init-gen.json"
    out_path="${out_dir}/${revision_no_family}-revise-one-step-${generator_no_family}-${split}.json"
    if [ $index -eq $task_id ]; then
      if [ ! -f $out_path ]; then
        echo "Running combination: generator=$generator, revision_model=$revision_model"
        ./inference_pipeline/run_single_revision_one_step.sh $path_to_init_response_generator $revision_model $out_path
        exit 0
      fi
    fi
    index=$((index + 1))
  done
done

