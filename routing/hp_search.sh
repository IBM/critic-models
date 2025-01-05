#!/bin/bash

# Define hyperparameter grid
learning_rates=(2e-5 3e-5)
batch_sizes=(4 8)
num_epochs=(3 5)
weight_decays=(0.01 0.1)
lora_rs=(8 16)
lora_alphas=(16 32)
lora_dropouts=(0.1 0.2)
seeds=(42 123)

config="utils/config_for_routing.json"
df="_output/data_for_critic_routing.csv"
model_name="google/gemma-2-2b-it"
# Output results file
results_file="_output/binary_classification_results.csv"

# Ensure results file exists with a header
if [ ! -f "$results_file" ]; then
  echo "model_name,learning_rate,batch_size,num_epochs,weight_decay,lora_r,lora_alpha,lora_dropout,seed,f1,accuracy" > "$results_file"
fi

for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for epochs in "${num_epochs[@]}"; do
      for wd in "${weight_decays[@]}"; do
        for r in "${lora_rs[@]}"; do
          for alpha in "${lora_alphas[@]}"; do
            for dropout in "${lora_dropouts[@]}"; do
              for seed in "${seeds[@]}"; do
                job_name="${lr}lr_${bs}bs_${epochs}epochs_${wd}wd_${r}r_${alpha}alpha_${dropout}dropout_${seed}seed"
                sbatch --mem=40gb -c2 --time=2:0:0 --gres=gpu:1,vmem:40g --error=${job_name}/error_log_job%A.txt \
                  --output=${job_name}/output_log_job%A.txt --job-name=${job_name} --mail-user=gili.lior@mail.huji.ac.il \
                  --mail-type=ALL "./routing/run_training.sh" $config $df $model_name $lr $bs $epochs $wd $r $alpha $dropout $seed $results_file
                break
              done
              break
            done
            break
          done
          break
        done
        break
      done
      break
    done
    break
  done
  break
done

