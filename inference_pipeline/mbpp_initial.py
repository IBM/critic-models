from inference_pipeline.generate_initial_response import InitialResponse
from datasets import load_dataset
from argparse import ArgumentParser


class MbppInitial(InitialResponse):
    def load_data(self, data_path):
        data = load_dataset(data_path, split=self.split)
        return data

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for sample in self.data:
            prompt = sample["text"]
            tests = sample["tests"]
            concat_tests = "\n".join(tests)
            msg = f"{prompt}.\nYour code should satisfy these tests:\n{concat_tests}"
            to_predict.append(
                [
                    {"role": "user", "content": msg}
                ]
            )
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts


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
