from argparse import ArgumentParser
from inference_pipeline.base_infer import InferenceBase
from tqdm import tqdm
from inference_pipeline.generate_critic import CriticGen
import json

class RevisionWithExternalFeedbackGen(InferenceBase):

    def __init__(self, model, base_model_for_lora, data_path, revision_prompt):
        super().__init__(model, base_model_for_lora, data_path)
        self.revision_prompt = revision_prompt

        # load the original responses using CriticGen.INITIAL_RESPONSES_PATH_KEY
        path_to_responses = self.data[CriticGen.INITIAL_RESPONSES_PATH_KEY]
        self.original_responses = super().load_data(path_to_responses)
        self.data = super().load_data(data_path)


    def get_key_in_out_dict(self):
        return "revisions"
    
    def load_data(self, data_path):
        with open(data_path, 'rt') as f:
            data = json.load(f)
        return data

    def get_data_for_inference(self):
        to_predict = []
        ordered_prompts = []
        for prompt in tqdm(self.data):
            conversation = []
            for msg in self.original_responses[prompt]: # get the original task conversation (user inst + model response)
                conversation.append(msg)
                if msg['role'] == 'user':
                    inst_w_format = msg['content']
                if msg['role'] == 'assistant':
                    break
            critic = self.data[prompt][-1]["content"] # get the critic's response
            sample = eval(f"'{self.revision_prompt}'").format(critic=critic, original_instruction=inst_w_format)
            conversation.append(
                {"role": "user", "content": sample}
            )
            to_predict.append(conversation)
            ordered_prompts.append(prompt)
        return to_predict, ordered_prompts

    def get_out_dict_format(self):
        out_dict = {
            "revision_prompt": self.revision_prompt,
            "generator_model": self.model_name,
            "critics": self.data_path
        }
        return out_dict



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--critics", type=str, required=True,
                        help="Path to a json file with critic")
    parser.add_argument("--revision_prompt", type=str, required=True,
                        help="the prompt used for revision")
    parser.add_argument("--generator_model", type=str, required=True,
                        help="the model to perform the revision (generating after revisions) with")
    parser.add_argument("--base_model_for_lora", default=None,
                        help="if inference with lora trained model, provide the base model it was trained from")
    parser.add_argument("--out_path", type=str, required=True,
                        help="out path to save the revised responses")

    args = parser.parse_args()
    infer_model = RevisionWithExternalFeedbackGen(args.generator_model, args.base_model_for_lora, args.critics, args.revision_prompt)
    infer_model.predict(args.out_path)