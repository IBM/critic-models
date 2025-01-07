from routing.base_learning_class import LLM_Classifier, multi_class_eval
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import re


class MultiCriticClassifier(LLM_Classifier):
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
            dataset.append({"sample_text": sample, "initial_response": model_response,
                            "best_init_generation_model": generator_model,
                            "revision_scores": np.array(raw_scores[row["best_init_generation_model"]]),
                            "best_revised_scores": np.max(raw_scores[row["best_init_generation_model"]]),
                            "labels": int(row["best_critic"]),
                            })
        if counter > 0:
            print(f"skip {counter} samples")
        dataset = Dataset.from_list(dataset)
        return dataset

    def unique_class_eval(self, labels, predictions):
        test_set = self.dataset["test"]
        test_set = test_set.select(range(100))
        return multi_class_eval(labels, predictions, test_set)
