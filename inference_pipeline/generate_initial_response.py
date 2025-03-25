import os.path
from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase
from datasets import load_dataset, load_from_disk


class InitialResponse(InferenceBase):
    # FORMAT_ANSWER = "Provide your answer in the following format: [START] <answer> [END]"
    FORMAT_ANSWER = ""

    def __init__(self, model, base_model_for_lora, data_path, split, tasks_key):
        self.split = split
        self.tasks_key = tasks_key
        super().__init__(model, base_model_for_lora, data_path)
        
    def load_data(self, data_path):
        if os.path.exists(data_path):
            decomposition_ds = load_from_disk(data_path)[self.split]
        else:
            decomposition_ds = load_dataset(data_path, split=self.split)

        # Load and filter the original dataset
        orig_ds = load_dataset("lmsys/lmsys-chat-1m", split="train")
        conversation_ids = set(decomposition_ds["conversation_id"])
        orig_ds_filtered = orig_ds.filter(lambda x: x['conversation_id'] in conversation_ids)

        def leave_only_first_request(example):
            example["conversation"] = example["conversation"][0]["content"]
            return example

        # Keep only the first request in each conversation
        orig_ds_cleaned = orig_ds_filtered.map(leave_only_first_request)
        orig_ds_cleaned = orig_ds_cleaned.rename_column("conversation", "task")

        # Convert decomposition dataset into a dictionary for fast lookup
        decomposition_dict = {row["conversation_id"]: row for row in decomposition_ds}

        # Merge decomposition with original dataset
        def merge_examples(example):
            match = decomposition_dict.get(example["conversation_id"], {})
            return {**example, **match}

        merged_dataset = orig_ds_cleaned.map(merge_examples)

        return merged_dataset

    def get_key_in_out_dict(self):
        return "initial_responses"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in self.data["task"]:
            to_predict.append(
                [
                    {"role": "user", "content": prompt + " " + self.FORMAT_ANSWER}
                ]
            )
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "format_answer": self.FORMAT_ANSWER,
            "generator_model": self.model_name,
            "data_path": self.data_path,
            "split": self.split
        }
        return out_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="path to dataset to generate predictions for")
    parser.add_argument("--model", required=True,
                        help="path to model to generate predictions with")
    parser.add_argument("--base_model_for_lora", default=None,
                        help="if inference with lora trained model, provide the base model it was trained from")
    parser.add_argument("--out_path", required=True,
                        help="path to json file to save predictions to")
    parser.add_argument("--split", required=True, choices=['train', 'validation', 'test'])
    parser.add_argument("--tasks_key", required=True, help="the tasks column name")

    args = parser.parse_args()
    inference_model = InitialResponse(args.model, args.base_model_for_lora, args.dataset, args.split, args.tasks_key)
    inference_model.predict(args.out_path)