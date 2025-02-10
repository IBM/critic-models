
from argparse import ArgumentParser
from datasets import load_from_disk
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from utils.filter_data_consts import FILTER_PROMPT
import os
import re
import numpy as np
import concurrent.futures
from tqdm import tqdm
import json
from openai import OpenAI

WATSONX_API_ENDPOINT = "https://us-south.ml.cloud.ibm.com"
ERROR_SCORE = "ERR"


class BaseDataset:

    def __init__(self, name_or_path):
        self.name_or_path = name_or_path
        self.data = self.load_data(name_or_path)

    def get_name_or_path(self):
        return self.name_or_path

    def load_data(self, name_or_path):
        raise NotImplementedError

    def get_tasks_list(self):
        raise NotImplementedError


class ConstrainedGenerationClassification:
    def __init__(self, data: BaseDataset, model_name, max_new_tokens):
        self.dataset_name = data.get_name_or_path()
        self.data = data
        self.short_model_name = model_name
        self.model_name = self.get_model_name_in_server(model_name)
        self.credentials = self.get_credentials()
        self.max_new_tokens = max_new_tokens
        self.generation_params = self.get_generation_params()

    def get_name(self):
        return f"constrained-gen-pos-score"

    def get_out_path(self, out_dir):
        path = os.path.join(out_dir, f"{self.get_name()}-{self.short_model_name}.json")
        print(f"output path at {path}")
        return path

    def infer(self, out_dir):
        answers = {}
        out_path = self.get_out_path(out_dir)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.get_max_workers())
        future_to_task = {}

        if os.path.exists(out_path):
            with open(out_path, 'rt') as f:
                answers = json.load(f)

        feedback_file = open(out_path, 'wt')
        for task in set(self.data.get_tasks_list()):
            if task in answers:
                continue
            future_to_task[pool.submit(self._infer, task)] = task

        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(future_to_task)):
            task = future_to_task[future]
            feedback_dict = future.result()
            answers[task] = feedback_dict
            self.redump_json(feedback_file, answers)

        pool.shutdown(wait=True)
        self.redump_json(feedback_file, answers)
        feedback_file.close()

    @staticmethod
    def redump_json(feedback_file, answers):
        str_feedback_dict = json.dumps(answers, indent=2)
        feedback_file.seek(0)
        feedback_file.write(str_feedback_dict)

    def _infer(self, task):
        message = FILTER_PROMPT.format(request=task)
        answer = self.get_answer(message)
        generated_text = answer["results"][0]["generated_text"]
        generated_tokens = answer["results"][0]["generated_tokens"]
        pos_score = self.calc_score(generated_tokens)
        return {"answer": generated_text, "pos_score": pos_score}

    def get_answer(self, message):
        raise NotImplementedError

    def get_credentials(self):
        creds = Credentials(api_key=self.get_api_key(), url=self.get_api_endpoint())
        return creds

    def get_generation_params(self):
        generate_params = {
            GenParams.MAX_NEW_TOKENS: self.max_new_tokens,
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
            "return_options": {"input_tokens": True, "generated_tokens": True, "token_logprobs": True,
                               "top_n_tokens": 5}
        }
        return generate_params

    @staticmethod
    def get_model_name_in_server(model_name):
        if model_name == 'llama3-70b':
            full_name = 'meta-llama/llama-3-70b-instruct'
        elif model_name == 'llama3.1-70b':
            full_name = 'meta-llama/llama-3-1-70b-instruct'
        elif model_name == 'llama3.3-70b':
            full_name = 'meta-llama/llama-3-3-70b-instruct'
        elif model_name == 'llama3-405b':
            full_name = 'meta-llama/llama-3-405b-instruct'
        elif model_name == 'llama3.1-405b':
            full_name = 'meta-llama/llama-3-1-405b-instruct-fp8'
        elif model_name == 'llama3.1-8b':
            full_name = 'meta-llama/Llama-3.1-8B-Instruct'
        elif model_name == 'qwen2.5-72b':
            full_name = 'Qwen/Qwen2.5-72B-Instruct'
        elif model_name == 'deepseek-v3':
            full_name = 'deepseek-ai/DeepSeek-V3'
        elif model_name == 'mistral-large':
            full_name = 'mistralai/mistral-large-instruct-2407'
        else:
            raise RuntimeError(f"model unknown {model_name}")
        return full_name

    def get_api_key(self):
        raise NotImplementedError

    def get_api_endpoint(self):
        raise NotImplementedError

    def get_max_workers(self):
        raise NotImplementedError

    @staticmethod
    def calc_score(token_preds: list[dict]):
        num_tokens_to_check = 5
        min_probability_mass = 0.0001
        for i in range(min(num_tokens_to_check, len(token_preds))):
            try:
                pos_probs, neg_probs = ConstrainedGenerationClassification.get_pos_neg_probs(token_logprobs_obj=token_preds[i]["top_tokens"])
                if pos_probs or neg_probs:
                    sum_probs = sum(pos_probs) + sum(neg_probs)
                    if sum_probs > min_probability_mass:
                        return sum(pos_probs) / sum_probs
            except:
                pass
        return ERROR_SCORE

    @staticmethod
    def get_pos_neg_probs(token_logprobs_obj):
        pos_and_neg_probs = []
        for class_name in ["yes", "no"]:
            # We need to capture different variants of model behavior and tokenizers, for example with opening space,
            # punctuation etc. but avoid longer words that contain the class name.
            # For example, for class "yes" we would capture "YES," and " Yes" but not "yesterday".
            name_regex = re.compile(
                rf"(\W|Ġ|_)*{class_name}(\W|Ġ|_)*", flags=re.IGNORECASE
            )
            class_probs = [
                np.exp(d["logprob"])
                for d in token_logprobs_obj
                if name_regex.fullmatch(d["text"])
            ]
            pos_and_neg_probs.append(class_probs)
        return pos_and_neg_probs



class ConstrainedGenerationClassificationBam(ConstrainedGenerationClassification):
    BAM_API_ENDPOINT = "https://bam-api.res.ibm.com"
    BAM_API_KEY_VAR_NAME = "GENAI_KEY"
    MAX_WORKERS = 10

    def __init__(self, data: BaseDataset, model_name, max_new_tokens):
        super().__init__(data, model_name, max_new_tokens)
        from genai.client import Client
        self.client = Client(credentials=self.credentials)

    def get_credentials(self):
        from genai.credentials import Credentials
        creds = Credentials(api_key=self.get_api_key(), api_endpoint=self.get_api_endpoint())
        return creds

    def get_answer(self, message):
        response = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=message,
            parameters=self.generation_params,
        )
        answer = {"results": []}
        for resp in response:
            top_logprobs_response = resp.results[0].generated_tokens
            output = [
                {
                    "top_tokens": [
                        {"text": obj.text, "logprob": np.exp(obj.logprob) if obj.logprob is not None else 0}
                        for obj in generated_token.top_tokens
                    ]
                }
                for generated_token in top_logprobs_response
            ]
            answer['results'].append({"generated_tokens": output, "generated_text": resp.results[0].generated_text})
            break
        return answer

    def get_api_key(self):
        return os.environ.get(self.BAM_API_KEY_VAR_NAME)

    def get_api_endpoint(self):
        return self.BAM_API_ENDPOINT

    def get_max_workers(self):
        return self.MAX_WORKERS

class ConstrainedGenerationClassificationWMV(ConstrainedGenerationClassification):
    WMV_API_ENDPOINT = "https://us-south.ml.cloud.ibm.com"
    WMV_API_KEY_VAR_NAME = "WATSONX_KEY"
    MAX_WORKERS = 4
    WATSONX_PROJECT_ID_VAR_NAME = "WATSONX_PROJECT_ID"

    def __init__(self, data: BaseDataset, model_name, max_new_tokens):
        super().__init__(data, model_name, max_new_tokens)
        self.model_inference = ModelInference(
            model_id=self.model_name,
            params=self.generation_params,
            credentials=self.credentials,
            project_id=os.environ.get(self.WATSONX_PROJECT_ID_VAR_NAME),
        )

    def get_answer(self, message):
        answer = self.model_inference.generate(message)
        return answer

    def get_api_key(self):
        return os.environ.get(self.WMV_API_KEY_VAR_NAME)

    def get_api_endpoint(self):
        return self.WMV_API_ENDPOINT

    def get_max_workers(self):
        return self.MAX_WORKERS


class ConstrainedGenerationClassificationRITS(ConstrainedGenerationClassification):
    RITS_API_ENDPOINT = "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{}/v1"
    RITS_API_KEY_VAR_NAME = "RITS_API_KEY"
    MAX_WORKERS = 20

    def __init__(self, data: BaseDataset, model_name, max_new_tokens):
        self.model_name_for_endpoint = self.get_model_name_in_server(model_name).split(
            "/")[-1].lower().replace("v0.1", "v01").replace(".", "-")
        self.client = OpenAI(api_key=self.get_api_key(),
                             base_url=self.get_api_endpoint().format(self.model_name_for_endpoint))
        super().__init__(data, model_name, max_new_tokens)

    def get_generation_params(self):
        gen_params = super().get_generation_params()
        gen_params.pop('decoding_method')
        gen_params['temperature'] = 0
        gen_params['extra_headers'] = {"RITS_API_KEY": self.get_api_key()}
        if self.client.base_url.host == "api.openai.com":
            gen_params["max_completion_tokens"] = gen_params.pop("max_new_tokens", None)
        else:
            gen_params['max_tokens'] = gen_params.pop("max_new_tokens", None)

        if 'return_options' in gen_params:
            gen_params['top_logprobs'] = gen_params['return_options'].get("top_n_tokens")
            gen_params['logprobs'] = gen_params['return_options'].get("token_logprobs") is True
            gen_params.pop("return_options")
        return gen_params

    def get_answer(self, message):
        if type(message) is str:
            message = [{'role': 'user', 'content': message}]
        completion = self.client.chat.completions.create(
            messages=message,
            model=self.model_name,
            **self.generation_params
        )
        answer = {'results': []}
        generated_text = completion.choices[0].message.content
        top_logprobs_response = completion.choices[0].logprobs.content
        token_dicts = [
            {
                "top_tokens": [
                    {"text": obj.token, "logprob": obj.logprob}
                    for obj in generated_token.top_logprobs
                ]
            }
            for generated_token in top_logprobs_response
        ]
        answer['results'].append({"generated_tokens": token_dicts, "generated_text": generated_text})
        return answer

    def get_api_key(self):
        return os.environ.get(self.RITS_API_KEY_VAR_NAME)

    def get_api_endpoint(self):
        return self.RITS_API_ENDPOINT

    def get_max_workers(self):
        return self.MAX_WORKERS


class ArenaDataset(BaseDataset):
    def __init__(self, name_or_path):
        super().__init__(name_or_path)

    def load_data(self, name_or_path):
        data = load_from_disk(name_or_path)
        return data

    def get_tasks_list(self):
        return self.data["task"]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out_dir")
    parser.add_argument("--platform", choices=['bam', 'wmv', 'rits'])
    parser.add_argument("--model_name")
    args = parser.parse_args()
    dataset = ArenaDataset(args.data)
    if args.platform == "bam":
        classifier = ConstrainedGenerationClassificationBam(dataset, args.model_name, max_new_tokens=5)
    elif args.platform == "wmv":
        classifier = ConstrainedGenerationClassificationWMV(dataset, args.model_name, max_new_tokens=5)
    else:  # args.platform == "rits":
        classifier = ConstrainedGenerationClassificationRITS(dataset, args.model_name, max_new_tokens=5)
    classifier.infer(args.out_dir)