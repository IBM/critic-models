from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase, LoRARequest
from utils.utils import process_response
import json
from tqdm import tqdm
import os

class CriticAndRefineGen(InferenceBase):
    INITIAL_RESPONSES_PATH_KEY = "refined_responses"

    def __init__(self, model, base_model_for_lora, data_path, critic_prompt):
        super().__init__(model, base_model_for_lora, data_path)
        self.critic_prompt = critic_prompt

    def get_key_in_out_dict(self):
        return "revisions"

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in tqdm(self.data):
            for msg in self.data[prompt]:
                if msg['role'] == 'assistant':
                    ai_response = msg['content']
                    break
            ai_response, _ = process_response(ai_response)

            sample = eval(f"'{self.critic_prompt}'").format(ai_response=ai_response, instruction=prompt)
            to_predict.append([
                {"role": "user", "content": sample}
            ])
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_data_for_refinement(self, to_predict, ordered_prompts):
        for i in range(len(to_predict)):
            to_predict[i].append({"role": "user", "content": "Using the task description and the feedback you provided, craft a new response that aligns with the task requirements. Be aware of the strengths and the pitfalls of the reference output, while you generate the best final output possible that fulfills the task requirements:\n\"{TASK}\".\n\nAnswer:".format(TASK=ordered_prompts[i])})
        return to_predict
    def predict(self, out_path):
        to_predict, ordered_prompts = self.get_data_for_inference()
        pred_with_outputs_first_step = self.get_predictions(to_predict, ordered_prompts)
        second_step_to_predict = self.get_data_for_refinement(pred_with_outputs_first_step, ordered_prompts)
        second_step_out = self.get_predictions(second_step_to_predict, ordered_prompts)
        self.dump_output(ordered_prompts, out_path, second_step_out)


    def get_out_dict_format(self):
        out_dict = {
            "critic_prompt": self.critic_prompt,
            "critic_model": self.model_name,
            self.INITIAL_RESPONSES_PATH_KEY: self.data_path
        }
        return out_dict



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--responses", type=str, required=True,
                        help="Path to a json file with responses")
    parser.add_argument("--critic_prompt", type=str, required=True,
                        help="the prompt used for critic")
    parser.add_argument("--critic_model", type=str, required=True,
                        help="the model to perform the critic with")
    parser.add_argument("--base_model_for_lora", default=None,
                        help="if inference with lora trained model, provide the base model it was trained from")
    parser.add_argument("--out_path", type=str, required=True,
                        help="out path to save the critic responses")

    args = parser.parse_args()
    infer_critics_model = CriticAndRefineGen(args.critic_model, args.base_model_for_lora, args.responses, args.critic_prompt)
    infer_critics_model.predict(args.out_path)