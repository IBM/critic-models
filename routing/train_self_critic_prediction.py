
from argparse import ArgumentParser
import json

import numpy as np
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

SPLIT = 'train'


def load_initial_response(json_data_for_model, sample):
    pred_key = json_data_for_model["predictions_key"]
    text = f"{json_data_for_model[pred_key][sample][0]['content']}\n\n{json_data_for_model[pred_key][sample][1]['content']}"
    return text


def load_all_models_jsons(config, generator_models):
    all_jsons = {}
    for model in generator_models:
        path = config["generation_init_path"].format(generation_init_dir=config["generation_init_dir"],
                                                     generator_model=model, split=SPLIT)
        with open(path, 'r') as f:
            json_data = json.load(f)
        all_jsons[model] = json_data
    return all_jsons

def generate_training_data(df, config):
    sorted_models = sorted(config["generator_models"])
    dataset = []
    all_models_jsons = load_all_models_jsons(config, sorted_models)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        sample = row["sample_text"]
        generator_model = row["best_init_generation_model"]
        generator_model = sorted_models[generator_model]
        model_response = load_initial_response(all_models_jsons[generator_model], sample)
        labels = np.zeros((2,), dtype=int)
        labels[int(row["is_self_critic_best"])] = 1
        dataset.append({"sample_text": sample, "initial_response": model_response,
                        "best_init_generation_model": generator_model,
                        "is_self_critic_best": row["is_self_critic_best"],
                        "labels": int(row["is_self_critic_best"]),
                        "no_critics_needed": row["no_critics_needed"],
                        "best_critic": row["best_critic"]
                        })
    dataset = Dataset.from_list(dataset)
    return dataset


def main(path_to_config, path_to_df, model_name):
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    df = pd.read_csv(path_to_df)[:1000]
    dataset = generate_training_data(df, config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, device_map='auto', torch_dtype='bfloat16')  # Binary classification

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        r=8,  # Low-rank update matrix
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Target specific layers (adjust based on model architecture)
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none"  # Freeze biases
    )
    model = get_peft_model(base_model, lora_config)

    # Assuming `dataset` is your Dataset object with "text" and "label" columns
    # Shuffle and split the dataset into train and test
    dataset = dataset.shuffle(seed=42)
    train_test_split = dataset.train_test_split(test_size=0.2)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["initial_response"], truncation=True, padding="longest", max_length=2048
        )

    tokenized_datasets = train_test_split.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        fp16=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("gemma2-2b_binary_classifier")
    tokenizer.save_pretrained("gemma2-2b_binary_classifier")

    # Inference on new data
    new_texts = tokenized_datasets["test"]["initial_response"]
    inputs = tokenizer(new_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # calculate accuracy, f1, etc.
    accuracy = (predictions == tokenized_datasets["test"]["label"]).float().mean
    f1 = f1_score(tokenized_datasets["test"]["label"], predictions, average='binary')  # Calculate F1 score

    baselines_accuracy_all_0 = (tokenized_datasets["test"]["label"] == 0).float().mean()
    baselines_accuracy_all_1 = (tokenized_datasets["test"]["label"] == 1).float().mean()
    print(f"Accuracy: {accuracy}")
    print(f"Baseline accuracy all 0: {baselines_accuracy_all_0}")
    print(f"Baseline accuracy all 1: {baselines_accuracy_all_1}")
    baseline_f1_all_0 = f1_score(tokenized_datasets["test"]["label"], [0] * len(tokenized_datasets["test"]["label"]), average='binary')
    print(f"F1 Score: {f1}")
    baseline_f1_all_1 = f1_score(tokenized_datasets["test"]["label"], [1] * len(tokenized_datasets["test"]["label"]), average='binary')
    print(f"Baseline F1 all 0: {baseline_f1_all_0}")
    print(f"Baseline F1 all 1: {baseline_f1_all_1}")




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--df', type=str, help='path to dataframe with labels')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--model_name', type=str, help='name of the model to use')
    args = parser.parse_args()
    main(args.config, args.df, args.model_name)
