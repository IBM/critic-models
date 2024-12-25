import os.path
from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase
from datasets import load_dataset, load_from_disk


class InitialResponse(InferenceBase):
    FORMAT_ANSWER = "Provide your answer in the following format: [START] <answer> [END]"

    def __init__(self, model, base_model_for_lora, data_path, split, tasks_key):
        self.split = split
        self.tasks_key = tasks_key
        super().__init__(model, base_model_for_lora, data_path)
        
    def load_data(self, data_path):
        if os.path.exists(data_path):
            data = load_from_disk(data_path)[self.split][self.tasks_key]
        else:
            data = load_dataset(data_path, split=self.split)[self.tasks_key]
        return data

    def get_key_in_out_dict(self):
        return "initial_responses"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in self.data:
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