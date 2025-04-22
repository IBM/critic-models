from inference_pipeline.base_infer import InferenceBase
from datasets import load_dataset
from argparse import ArgumentParser


REVISION_PROMPT = """
Reflect on your previous response: Did you fully address all aspects of the instruction? Could anything be clearer, more concise, or more relevant? If so, revise your response. If not, remain silent.
Original instruction: {instruction}
Respond only with your revised answer — no explanations.
""".strip()

class MultipleIterationsInContext(InferenceBase):
    def __init__(self, model, base_model_for_lora, data_path, num_iterations):
        self.num_iterations = num_iterations
        super().__init__(model, base_model_for_lora, data_path)

    def load_data(self, data_path):
        dataset = load_dataset(data_path, split="all")
        return dataset

    def get_key_in_out_dict(self):
        return "multiple_revisions_in_context"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in self.data["prompt"]:
            to_predict.append(
                [
                    {"role": "user", "content": prompt}
                ]
            )
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "generator_model": self.model_name,
            "data_path": self.data_path,
        }
        return out_dict

    def get_predictions(self, to_predict, ordered_prompts):
        print("processing prompts...")

        print("generating responses...")
        outputs = self.inference_model.chat(messages=to_predict, sampling_params=self.sampling_params, use_tqdm=True)
        for j in range(self.num_iterations):
            for i, prompt in enumerate(ordered_prompts):
                response = outputs[i].outputs[0].text
                to_predict[i].append({"role": "assistant", "content": response})
                to_predict[i].append({"role": "user", "content": REVISION_PROMPT.format(instruction=prompt)})
            outputs = self.inference_model.chat(messages=to_predict, sampling_params=self.sampling_params,
                                                use_tqdm=True)

        return to_predict

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

    args = parser.parse_args()
    inference_model = MultipleIterationsInContext(args.model, args.base_model_for_lora, args.dataset, args.num_iterations)
    inference_model.predict(args.out_path)