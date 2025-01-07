
import json
import os
import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
import numpy as np
import random
import torch
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score
from transformers import DataCollatorWithPadding

MAX_LENGTH = 2048
SPLIT = 'train'


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    if labels.ndim == 2:
        labels = np.argmax(labels, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    if np.max(labels) > 1:
        f1 = f1_score(labels, predictions, average='micro')
    else:
        f1 = f1_score(labels, predictions, average='binary')
    print(f"Evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return {"accuracy": accuracy, "f1": f1}


class LLM_Classifier:

    def load_hf_model(self, model_name):
        raise NotImplementedError

    @staticmethod
    def set_random_seed(seed: int):
        """Set seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_peft_model(self, base_model):
        # Initialize LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias="none"
        )
        model = get_peft_model(base_model, lora_config)
        return model

    def load_dataset(self):
        df = pd.read_csv(self.path_to_df)
        dataset = self.generate_training_data(df, self.config)
        dataset = dataset.shuffle(seed=self.seed).train_test_split(test_size=0.2)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["initial_response"], truncation=True
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset


    def generate_training_data(self, df, config):
        raise NotImplementedError

    @staticmethod
    def load_all_models_jsons(config, generator_models):
        all_jsons = {}
        for model in generator_models:
            path = config["generation_init_path"].format(generation_init_dir=config["generation_init_dir"],
                                                         generator_model=model, split=SPLIT)
            with open(path, 'r') as f:
                json_data = json.load(f)
            all_jsons[model] = json_data
        return all_jsons

    @staticmethod
    def load_initial_response(json_data_for_model, sample):
        pred_key = json_data_for_model["predictions_key"]
        text = f"{json_data_for_model[pred_key][sample][0]['content']}\n\n{json_data_for_model[pred_key][sample][1]['content']}"
        return text

    def run_all(self):
        trainer = self.train_model()
        self.evaluate_model(trainer)

    def __init__(self, path_to_config, path_to_df, model_name, learning_rate, batch_size, num_epochs,
                 weight_decay, lora_r, lora_alpha, lora_dropout, results_file, seed):
        # Set seed for reproducibility
        set_seed(seed)
        self.set_random_seed(seed)
        self.seed = seed

        # Load config and data
        with open(path_to_config, 'r') as f:
            self.config = json.load(f)

        self.path_to_df = path_to_df
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.results_file = results_file

        hf_model = self.load_hf_model(model_name)
        self.hf_model = self.load_peft_model(hf_model)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=MAX_LENGTH)
        self.dataset = self.load_dataset()

    def get_training_args(self):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=1000,  # Save every 1000 steps
            eval_steps=500,  # Evaluate every 500 steps
            logging_dir="./logs",
            logging_steps=100,  # Log every 100 steps
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=8 // self.batch_size,
            per_device_eval_batch_size=4,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True
        )
        return training_args

    def train_model(self):
        # Initialize the Trainer
        training_args = self.get_training_args()
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, max_length=MAX_LENGTH)
        trainer = Trainer(
            model=self.hf_model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
        self.evaluate_model(trainer)

        # Train the model
        trainer.train()
        return trainer

    def unique_class_eval(self, labels, predictions):
        raise NotImplementedError

    def evaluate_model(self, trainer):

        # Evaluate the model
        predictions, labels, _ = trainer.predict(self.dataset["test"])
        predictions = predictions.argmax(axis=1)
        res = self.unique_class_eval(np.array(labels), np.array(predictions))

        # Print model name and hyperparameters
        print(f"Model: {self.model_name}")
        print(
            f"Hyperparameters: Learning Rate={self.learning_rate}, Batch Size={self.batch_size}, "
            f"Num Epochs={self.num_epochs}, Weight Decay={self.weight_decay}, LoRA R={self.lora_r}, "
            f"LoRA Alpha={self.lora_alpha}, LoRA Dropout={self.lora_dropout}")
        for key, value in res.items():
            print(f"{key}: {value}")


        # Save results to a file
        hyperparams = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "seed": self.seed,
        }
        self.save_results(hyperparams, res)

    def save_results(self, hyperparams, res):
        """Save the results to a CSV file for comparison."""
        results = {
            "model_name": self.model_name,
            **hyperparams,
            **res
        }
        df = pd.DataFrame([results])

        # Append results to the file or create a new one if it doesn't exist
        if not os.path.exists(self.results_file):
            df.to_csv(self.results_file, index=False)
        else:
            df.to_csv(self.results_file, mode='a', header=False, index=False)