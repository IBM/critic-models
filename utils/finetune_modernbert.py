from argparse import ArgumentParser
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score
import json

DATA_PATH = "_output/dataset.csv"

def main(model_name, epochs, lr, batch_size):
    dataset = pd.read_csv(DATA_PATH)
    # dataset = dataset[dataset["small_model_perfect"]!=1] # remove tasks where small model is perfect
    labels = []
    for _, row in dataset.iterrows():
        if row["small_model_perfect"]:
            labels.append(0)
        elif row["both_revision_and_bigger_not_better"]:
            labels.append(1)
        elif row["revision_with_gemma_is_best"]:
            labels.append(2)
        else:  # zero shot with big model is best
            labels.append(3)
    dataset["labels"] = labels
    ds = Dataset.from_pandas(dataset)
    ds_split = ds.train_test_split(test_size=0.2, seed=42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch['sample'], padding=True, truncation=True)

    tokenized_dataset = ds_split.map(tokenize, batched=True, batch_size=-1)

    # prepare labels
    labels = sorted(set(tokenized_dataset["train"]["labels"]))
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": label2id[x["labels"]]}, batched=False)

    # Download the model from huggingface.co/models
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, label2id=label2id, id2label=id2label, device_map="auto"
    )

    # Metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        score = f1_score(labels, predictions, average="weighted")
        return {"f1": score}

    # Training arguments with CUDA acceleration
    training_args = TrainingArguments(
        output_dir="ModernBERT-domain-classifier",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        learning_rate=lr,
        num_train_epochs=epochs,
        bf16=not torch.cuda.is_available(),  # Use bfloat16 if supported
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        use_mps_device=not torch.cuda.is_available(),
        metric_for_best_model="f1",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),  # Enable fp16 if CUDA is available
    )

    # Trainer with CUDA support
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    predictions = trainer.predict(tokenized_dataset["test"])
    max_pred = np.argmax(predictions.predictions, axis=1)
    json_out = {}
    for i, sample in enumerate(tokenized_dataset["test"].iter(batch_size=1)):
        json_out[sample["sample"][0]] = int(max_pred[i])

    out_path = f"_output/modernbert_predictions_v2/{batch_size}batch_size_{epochs}epochs_{lr}lr.json"
    with open(out_path, "w") as f:
        pretty_json = json.dumps(json_out, indent=2)
        f.write(pretty_json)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()
    main(args.model, args.epochs, args.lr, args.batch_size)
