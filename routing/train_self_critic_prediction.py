import random
from argparse import ArgumentParser
import json
import numpy as np
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
import os

SPLIT = 'train'


def set_random_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        try:
            model_response = load_initial_response(all_models_jsons[generator_model], sample)
        except KeyError:
            print(f"Sample {sample} not found in model {generator_model}")
            continue
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    print(f"Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return {"accuracy": accuracy, "f1": f1}


def save_results(output_file, model_name, hyperparams, f1, accuracy):
    """Save the results to a CSV file for comparison."""
    results = {
        "model_name": model_name,
        **hyperparams,
        "f1": f1,
        "accuracy": accuracy,
    }
    df = pd.DataFrame([results])

    # Append results to the file or create a new one if it doesn't exist
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)


def main(path_to_config, path_to_df, model_name, learning_rate, batch_size, num_epochs, weight_decay, lora_r,
         lora_alpha, lora_dropout, results_file, seed):
    # Set seed for reproducibility
    set_seed(seed)
    set_random_seed(seed)

    # Load config and data
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    df = pd.read_csv(path_to_df)
    dataset = generate_training_data(df, config)
    dataset = dataset.shuffle(seed=seed).train_test_split(test_size=0.2)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, device_map='auto', torch_dtype='bfloat16'
    )

    # Initialize LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["initial_response"], truncation=True, padding="longest", max_length=2048
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=1000,  # Save every 1000 steps
        eval_steps=1000,  # Evaluate every 1000 steps
        logging_dir="./logs",
        logging_steps=100,  # Log every 100 steps
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8 // batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # # Save the model
    # model_name_for_save = model_name.split("/")[-1] + '_binary_classifier'
    # path_to_save = os.path.join(config["model_save_dir"], model_name_for_save)
    # model.save_pretrained(path_to_save)

    # Evaluate the model
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')

    # Print model name and hyperparameters
    print(f"Model: {model_name}")
    print(
        f"Hyperparameters: Learning Rate={learning_rate}, Batch Size={batch_size}, Num Epochs={num_epochs}, Weight Decay={weight_decay}, LoRA R={lora_r}, LoRA Alpha={lora_alpha}, LoRA Dropout={lora_dropout}")
    print(f"Final Accuracy: {accuracy}")
    print(f"Final F1 Score: {f1}")

    # Save results to a file
    hyperparams = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "seed": seed,
    }
    save_results(results_file, model_name, hyperparams, f1, accuracy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--df', type=str, required=True, help='Path to dataframe with labels')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, required=True, help='Weight decay for optimization')
    parser.add_argument('--lora_r', type=int, required=True, help='LoRA parameter r (rank)')
    parser.add_argument('--lora_alpha', type=int, required=True, help='LoRA parameter alpha (scaling factor)')
    parser.add_argument('--lora_dropout', type=float, required=True, help='LoRA parameter dropout rate')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to results file for saving metrics and hyperparameters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    main(
        args.config,
        args.df,
        args.model_name,
        args.learning_rate,
        args.batch_size,
        args.num_epochs,
        args.weight_decay,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.results_file,
        args.seed
    )
