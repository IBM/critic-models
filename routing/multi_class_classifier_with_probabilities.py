from routing.base_learning_class import LLM_Classifier
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import re


class MultiProbableCriticClassifier(LLM_Classifier):
    def load_hf_model(self, model_name):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=6, device_map='auto', torch_dtype='bfloat16'
        )
        return base_model

    def generate_training_data(self, df, config):
        sorted_models = sorted(config["generator_models"])
        dataset = []
        all_models_jsons = self.load_all_models_jsons(config, sorted_models)
        counter = 0
        for i, row in tqdm(df.iterrows(), total=len(df)):
            sample = row["sample_text"]
            generator_model = row["best_init_generation_model"]
            generator_model = sorted_models[generator_model]
            try:
                model_response = self.load_initial_response(all_models_jsons[generator_model], sample)
            except KeyError:
                counter += 1
                continue
            raw_scores = eval(re.sub(r"\s+", ", ", row['scores']))
            labels = raw_scores[row["best_init_generation_model"]]
            exp_x = np.exp(labels - np.max(labels))  # Subtract max for numerical stability
            labels = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            dataset.append({"sample_text": sample, "initial_response": model_response,
                            "best_init_generation_model": generator_model,
                            "revision_scores": np.array(raw_scores[row["best_init_generation_model"]]),
                            "best_revised_scores": np.max(raw_scores[row["best_init_generation_model"]]),
                            "best_critic": int(row["best_critic"]),
                            "labels": labels,
                            })
        if counter > 0:
            print(f"skip {counter} samples")
        dataset = Dataset.from_list(dataset)
        return dataset

    def unique_class_eval(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        test_set = self.dataset["test"]
        gold_scores = test_set["best_revised_scores"]
        revision_scores = np.array(test_set["revision_scores"])
        predicted_scores = revision_scores[np.arange(len(predictions)), predictions]
        init_scores = revision_scores[:, -1]
        distance_from_gold = np.mean(gold_scores - predicted_scores)
        improvement_over_init = np.mean(predicted_scores - init_scores)
        improvement_over_init_when_critic_is_needed = np.mean(predicted_scores[labels != 5] - init_scores[labels != 5])
        distance_from_gold_when_critic_is_needed = np.mean(gold_scores[labels != 5] - predicted_scores[labels != 5])

        return {"accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro,
                "distance_from_gold": distance_from_gold, "improvement_over_init": improvement_over_init,
                "improvement_over_init_when_critic_is_needed": improvement_over_init_when_critic_is_needed,
                "distance_from_gold_when_critic_is_needed": distance_from_gold_when_critic_is_needed}
