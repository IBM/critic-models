import torch
from transformers import LongformerTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import pandas as pd


# Load tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2)

# Load dataset (using IMDb for binary classification as an example)
dataset = load_dataset("imdb")

# convert df to dataset
path_to_df = "_output/data_for_critic_routing.csv"
df = pd.read_csv(path_to_df)
# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=4096):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row["tokenized"]
        label = row["is_self_critic_best"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def tokenize_function(examples):
    return tokenizer(
        examples["sample_text"],
        padding="max_length",
        truncation=True,
        max_length=4096
    )

df["tokenized"] = df.apply(tokenize_function, axis=1)
dataset = CustomDataset(df, tokenizer)


# Prepare the dataset for PyTorch
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # Number of epochs = 3
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
epochs = 3
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.update(1)

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.2f}")
