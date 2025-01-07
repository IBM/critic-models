from routing.base_learning_class import LLM_Classifier
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

class SelfCriticClassifier(LLM_Classifier):

    def unique_class_eval(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='binary')
        return {"accuracy": accuracy, "f1": f1}

    def load_hf_model(self, model_name):
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, device_map='auto', torch_dtype='bfloat16'
        )
        return base_model

    def generate_training_data(self, df, config):
        sorted_models = sorted(config["generator_models"])
        dataset = []
        all_models_jsons = self.load_all_models_jsons(config, sorted_models)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            sample = row["sample_text"]
            generator_model = row["best_init_generation_model"]
            generator_model = sorted_models[generator_model]
            try:
                model_response = self.load_initial_response(all_models_jsons[generator_model], sample)
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
