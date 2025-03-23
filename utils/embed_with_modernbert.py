import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import json


DATA_PATH = "_output/dataset.csv"
ZERO_SHOT_PATH = "/cs/snapless/gabis/gililior/wild-if-eval-code/model_predictions/Llama-3.1-8B-0shot-wild-if-eval.json"
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 32

# Load dataset
dataset = pd.read_csv(DATA_PATH)

# Convert to Hugging Face Dataset
ds = Dataset.from_pandas(dataset)

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "mps"
# model.to(device)
model.eval()  # Set model to evaluation mode


print("loading zero shot data...")
with open(ZERO_SHOT_PATH, 'rt') as f:
    zero_shot_data = json.load(f)
if "predictions_key" in zero_shot_data:
    zero_shot_data = zero_shot_data[zero_shot_data["predictions_key"]]


# Function to get sentence embeddings
def get_embeddings(batch):
    inputs = tokenizer(batch["sample"], padding="longest", truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings_batch = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use CLS token embedding

    return {"embeddings": embeddings_batch}


# Compute embeddings
print("Computing embeddings...")
data_with_embeddings = ds.map(get_embeddings, batched=True, batch_size=BATCH_SIZE)

# save embeddings
embeddings = np.concatenate(data_with_embeddings["embeddings"], axis=0)
np.save("_output/modernbert_embeddings_tasks.npy", embeddings)

# save also tasks list for ordering
tasks = data_with_embeddings["task"]
with open("_output/tasks_list.json", "w") as f:
    json.dump(tasks, f)

print("computing embeddings for tasks and outputs...")
all_concat = []
for task in data_with_embeddings["task"]:
    concatenated_task_and_output = f"{task}\n\n{zero_shot_data[task][-1]['content']}"
    if task == data_with_embeddings["task"][0]:
        print("example of concatenated task and output:")
        print(concatenated_task_and_output)
    all_concat.append(concatenated_task_and_output)

new_df = pd.DataFrame({"sample": all_concat, "only_task": data_with_embeddings["task"]})
new_ds = Dataset.from_pandas(new_df)
print("Computing embeddings...")
new_ds = new_ds.map(get_embeddings, batched=True, batch_size=BATCH_SIZE)
embeddings = np.concatenate(new_ds["embeddings"], axis=0)
np.save("_output/modernbert_embeddings_tasks_and_outputs.npy", embeddings)