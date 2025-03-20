from argparse import ArgumentParser
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import f1_score

batch_size=8

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

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize helper function
    def tokenize(batch):
        return tokenizer(batch['sample'], padding=True, truncation=True)

    tokenized_dataset = ds_split.map(tokenize, batched=True, remove_columns=["sample"], batch_size=-1)

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
        model_name, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )



    # Metric helper method
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
        return {"f1": float(score) if score == 1 else score}

    # Define training args
    training_args = TrainingArguments(
        output_dir="ModernBERT-domain-classifier",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size//2,
        learning_rate=5e-5,
        num_train_epochs=5,
        bf16=True,  # bfloat16 training
        optim="adamw_torch_fused",  # improved optimizer
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        use_mps_device=True,
        metric_for_best_model="f1",
        # push to hub parameters
        push_to_hub=False,
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # {'train_runtime': 3642.7783, 'train_samples_per_second': 1.235, 'train_steps_per_second': 0.04, 'train_loss': 0.535627057634551, 'epoch': 5.0}

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    args = parser.parse_args()
    main(args.model)
