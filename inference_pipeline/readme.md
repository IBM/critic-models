
Running initial inference
```bash
python inference_pipeline/generate_initial_response.py --dataset /local/path/or/name/in/hub/ \
    --model HUGGINGFACE_GENERATOR_MODEL_NAME --out_path INIT_RESPONSE_OUT_PATH --split SPLIT --tasks_key task
```


Running Critic
```bash
python inference_pipeline/generate_critic.py \
  --critic_prompt "You are an assistant whose job is to help me perform tasks. I will give you an instruction and an AI assistant response. The instruction include some constraints to be followed by AI assistant while generating response. Your job is to provide feedback on the response. In your critic, address the strengths and weaknesses of the response, and identify which constraints are kept and which should be improved.\nInstruction: {instruction}\nAssistant Response: {ai_response}\n\nAnswer:" \
  --critic_model HUGGINGFACE_CRITIC_MODEL_NAME --responses INIT_RESPONSE_OUT_PATH --out_path CRITIC_OUT_PATH
```


Running Revision
```bash
python inference_pipeline/generate_revision_with_external_feedback.py \
  --revision_prompt "{critic}\n\nBased on the feedback above, generate a revised response for the original instruction: {original_instruction}\n\nAnswer:" \
  --critics CRITIC_OUT_PATH --generator_model HUGGINGFACE_GENERATOR_MODEL_NAME --out_path REVISION_OUT_PATH
```


Running evaluation - LLM as a judge over constraints (using RITS).
```bash
python inference_pipeline/llms_aaj_constraint_multiproc.py \
  --eval_model MODEL_NAME_IN_RITS --to_eval JSON_PATH_WITH_RESPONSES \
  --constraints JSON_PATH_WITH_DECOMPOSITION --out_dir OUT_DIR [--sample NUM_SAMPLES_TO_EVAL]
```
