
Generate embeddings representation. 
If embedding model is a decoder only, takes the last hidden state as representation. 
If using an encoder model, set the flag `--encoder_only`, and then will use sentence_transformers with the required embedding model.

```bash
python routing/represent_tasks.py \
  --initial_responses JSON_PATH_WITH_RESPONSES --decomposition JSON_PATH_WITH_DECOMPOSITION \
  --embedding_model EMBEDDING_MODEL_HUGGINGFACE_NAME --out_dir OUT_DIR [--encoder_only]
```


Generate visulizations and analysis over the embeddingd.
REPRESENTATION can be {tasks, input_and_output, outputs}, according to the embedding you run in the represent_tasks.py script.

```bash
python routing/visualize_data.py \
    --config utils/config_to_routing.json --split SPLIT \
    --representation REPRESENTATION --out_dir OUT_DIR
```