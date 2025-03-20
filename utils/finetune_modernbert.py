from argparse import ArgumentParser
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score

batch_size = 16
DATA_PATH = "_output/dataset.csv"

def main(model_name):
    dataset = pd.read_csv(DATA_PATH)
    dataset = dataset[dataset["small_model_perfect"]!=1] # remove tasks where small model is perfect
    labels = []
    for _, row in dataset.iterrows():
        if row["both_revision_and_bigger_not_better"]:
            labels.append(0)
        elif row["revision_with_gemma_is_best"]:
            labels.append(1)
        else:  # zero shot with big model is best
            labels.append(2)
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
        learning_rate=5e-5,
        num_train_epochs=10,
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

    predictions = trainer.predict(tokenized_dataset["test"])
    max_pred = np.argmax(predictions.predictions, axis=1)
    json_out = {}
    for i, sample in enumerate(tokenized_dataset["test"].iter(batch_size=1)):
        print(sample["sample"])
        print(max_pred[i])
        print(label2id)
        print(id2label)
        json_out[sample["sample"]] = max_pred[i]

    # Train the model
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    args = parser.parse_args()
    main(args.model)
