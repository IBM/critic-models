from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase
from utils.utils import process_response
from tqdm import tqdm


PROMPT = """
Your task is to generate text based on the provided task description. Additionally, you are given an example output from another model (the \"reference\"). Use the reference to identify elements that are well-executed and should be retained or adapted in your response, while also improving upon areas where it falls short or does not align with the task description. Focus on producing a result that best meets the requirements of the task and outperforms the reference. Do not provide any additional information.\n\nTask description={instruction}\n\nReference={ai_response}\n\nAnswer:
"""

class RefineGen(InferenceBase):
    INITIAL_RESPONSES_PATH_KEY = "refined_responses"

    def __init__(self, model, base_model_for_lora, data_path, refine_prompt):
        super().__init__(model, base_model_for_lora, data_path)
        self.refine_prompt = refine_prompt

    def get_key_in_out_dict(self):
        return "refine"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in tqdm(self.data):
            for msg in self.data[prompt]:
                if msg['role'] == 'assistant':
                    ai_response = msg['content']
                    break
            ai_response, _ = process_response(ai_response)

            sample = eval(f"'{self.refine_prompt}'").format(ai_response=ai_response, instruction=prompt)
            to_predict.append([
                {"role": "user", "content": sample}
            ])
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "refine_prompt": self.refine_prompt,
            "refine_model": self.model_name,
            self.INITIAL_RESPONSES_PATH_KEY: self.data_path
        }
        return out_dict



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--responses", type=str, required=True,
                        help="Path to a json file with responses")
    parser.add_argument("--refine_prompt", type=str, required=True,
                        help="the prompt used for revision")
    parser.add_argument("--refine_model", type=str, required=True,
                        help="the model to perform the revision with")
    parser.add_argument("--base_model_for_lora", default=None,
                        help="if inference with lora trained model, provide the base model it was trained from")
    parser.add_argument("--out_path", type=str, required=True,
                        help="out path to save the revision responses")

    args = parser.parse_args()
    infer_refine_model = RefineGen(args.refine_model, args.base_model_for_lora, args.responses, args.refine_prompt)
    infer_refine_model.predict(args.out_path)