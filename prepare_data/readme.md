This directory includes all the scripts used to curate our constrained generation dataset.

```bash
OUT_DIR_PATH='/some/path/to/save/data'
CLASSIFIER_MODEL='llama3-405b'  # the model used for classifying whether a task is a constrained generation task.
POS_SCORE_FILTERING_PERCENTILE=90  # how confident to you want the classifier model to be
DECOMPOSER_MODEL='llama3.1-70b'  # the model used for extracting the constraints of a task

# filter out non-english, code, etc.
python prepare_data/heuristic_filtering.py --dataset lmsys/lmsys-chat-1m --out_dir ${OUT_DIR_PATH}

# get score for each task - whether it is constrained generation or not
python prepare_data/classify_constrained_generation_tasks.py --out_dir ${OUT_DIR_PATH} \
  --data ${OUT_DIR_PATH}/lmsys-chat-1m-heuristic-filtered-train --platform WMV --model_name ${CLASSIFIER_MODEL}

python prepare_data/filter_tasks_given_pos_score.py --percentile $POS_SCORE_FILTERING_PERCENTILE \
  --out_dir ${OUT_DIR_PATH}/ --scores ${OUT_DIR_PATH}constrained-gen-pos-score-${CLASSIFIER_MODEL}.json

python prepare_data/decompose_tasks.py --out ${OUT_DIR_PATH} --model_name ${DECOMPOSER_MODEL} \
  --data ${OUT_DIR_PATH}/filtered_0.9percentile_0.93threshold.json --platform rits

python prepare_data/prepare_hf_dataset.py --decomposition ${OUT_DIR_PATH}/decomposition-${DECOMPOSER_MODEL}.json \
  --orig_dataset ${OUT_DIR_PATH}/lmsys-chat-1m-heuristic-filtered-train --ds_out_path ${OUT_DIR_PATH}/constrained-lmsys-chat-1m
```