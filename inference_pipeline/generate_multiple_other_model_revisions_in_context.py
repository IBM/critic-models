from inference_pipeline.generate_multiple_revisions_in_context import MultipleIterationsInContext
from datasets import load_dataset
from argparse import ArgumentParser
import json

class OtherModelMultRevisions(MultipleIterationsInContext):
    def __init__(self, model, base_model_for_lora, data_path, num_iterations, starting_index):
        self.starting_index = starting_index
        super().__init__(model, base_model_for_lora, data_path, num_iterations)

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            dataset = json.load(f)
        if "prediction_keys" in dataset:
            dataset = dataset["prediction_keys"]
        self.initial_model = dataset["generator_model"]
        for prompt in dataset:
            print(type(dataset[prompt]))
            print(dataset[prompt])
            break
        return dataset

    def get_key_in_out_dict(self):
        return "multiple_other_model_revisions"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in self.data:
            conversation = []
            count_assistant_messages = -1
            for msg in self.data[prompt]:
                print(self.data[prompt])
                if msg["role"] == "user":
                    conversation.append(msg)
                else: # assistant
                    count_assistant_messages += 1
                    if count_assistant_messages <= self.starting_index:
                        conversation.append(msg)
                    else:
                        break
            to_predict.append(conversation)
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "generator_model": self.model_name,
            "data_path": self.data_path,
            "initial_model": self.initial_model,
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
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="number of iterations to generate")
    parser.add_argument("--starting_index", type=int, default=0, help="index of the first revision to generate. if 0, the model starts from the 0 shot prediction")

    args = parser.parse_args()
    inference_model = OtherModelMultRevisions(args.model, args.base_model_for_lora, args.dataset, args.num_iterations, args.starting_index)
    inference_model.predict(args.out_path)