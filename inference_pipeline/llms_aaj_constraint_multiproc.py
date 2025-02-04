import os
from argparse import ArgumentParser
from prepare_data.classify_constrained_generation_tasks import ConstrainedGenerationClassificationRITS
from prepare_data.decompose_tasks import ArenaClassifiedData
from utils.eval_consts import PROMPT_EVAL
import random
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tqdm import tqdm


class ConstraintData(ArenaClassifiedData):
    def __init__(self, name_or_path, path_to_constraints, num_sample):
        super().__init__(name_or_path)
        constraints = self.load_data(path_to_constraints)
        self.constraints = {k.strip(): constraints[k] for k in constraints}
        self.num_sample = num_sample

    def load_data(self, name_or_path):
        with open(name_or_path, 'rt') as f:
            data = json.load(f)
        pred_key = data.get("predictions_key")
        if pred_key is not None:
            data = data[pred_key]
        return data

    def get_constraints(self, task):
        return self.constraints[task]

    def get_response(self, task):
        return self.data[task][-1]["content"]

    def get_tasks_list(self):
        list_of_tasks = list(self.data.keys())
        if self.num_sample > -1:
            tasks = sorted(list_of_tasks)
            list_of_tasks = random.Random(42).sample(tasks, k=self.num_sample)
        return list_of_tasks


class LLMJudgeConstraintsRITS(ConstrainedGenerationClassificationRITS):

    def get_name(self):
        return f"decomposition-evaluation"

    def get_out_path(self, out_dir):
        dataset_name = self.dataset_name.split(os.sep)[-1].replace(".json", "")
        path = os.path.join(out_dir, f"{self.get_name()}-{self.short_model_name}.{dataset_name}.json")
        print(f"output path at {path}")
        return path
    
    def dump_results(self, out_dir, all_scores):
        out_path_dump = self.get_out_path(out_dir)
        with open(out_path_dump, 'wt') as f:
            str_feedback_dict = json.dumps(all_scores, indent=2)
            f.write(str_feedback_dict)


def generate_parallel(obj, tasks):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in tasks:
        all_args[task] = {}
        response = obj.data.get_response(task)
        for atomic in obj.data.get_constraints(task):
            all_args[task][atomic] = (task, response, atomic)
            total += 1
    pbar = tqdm(total=total)
    for task in all_args:
        all_results[task] = {}
        for atomic in all_args[task]:
            arguments = all_args[task][atomic] + (api_key, base_url, model_name)
            all_results[task][atomic] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return all_results


def process_results(all_results):
    processed_results = {}

    for task in all_results:
        processed_results[task] = {"scores": {}, "explanations": {}}
        for atomic in all_results[task]:
            result = all_results[task][atomic].get()
            processed_results[task]["scores"][atomic] = result[0]
            processed_results[task]["explanations"][atomic] = result[1]
    return processed_results


def infer_local(task, response, atomic, api_key, base_url, model_name):
    message = PROMPT_EVAL.format(instruction=task, response=response, constraint=atomic) 
    message = [{'role': 'user', 'content': message}]
    client = OpenAI(api_key=api_key, base_url=base_url)

    gen_params = {
        GenParams.MAX_NEW_TOKENS: 5,
        "return_options": {"input_tokens": True, "generated_tokens": True, "token_logprobs": True, "top_n_tokens": 5},
        'temperature': 0,
        'extra_headers':  {"RITS_API_KEY": api_key}
    }

    if client.base_url.host == "api.openai.com":
        gen_params["max_completion_tokens"] = gen_params.pop("max_new_tokens", None)
    else:
        gen_params['max_tokens'] = gen_params.pop("max_new_tokens", None)

    if 'return_options' in gen_params:
        gen_params['top_logprobs'] = gen_params['return_options'].get("top_n_tokens")
        gen_params['logprobs'] = gen_params['return_options'].get("token_logprobs") is True
        gen_params.pop("return_options")
    
    completion = client.chat.completions.create(
            messages=message,
            model=model_name,
            **gen_params
        )
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

    pos_score = LLMJudgeConstraintsRITS.calc_score(token_dicts)
    results_for_atomic = pos_score
    explanations_for_atomic = {"tokens": token_dicts, "text": generated_text}

    return results_for_atomic, explanations_for_atomic


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--eval_model")
    parser.add_argument("--to_eval", help="path to the responses to evaluate")
    parser.add_argument("--constraints", help="path to the constraints decomposition json")
    parser.add_argument("--sample", type=int, default=-1,
                        help="specify how many samples to evaluate")
    parser.add_argument("--tasks_batch_size", type=int, default=200,
                        help="number of tasks to run inference on before saving")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print(f"\n\n=======\nEVALUATING {args.to_eval} WITH {args.eval_model}")
    dataset = ConstraintData(args.to_eval, args.constraints, args.sample)

    classifier = LLMJudgeConstraintsRITS(dataset, args.eval_model, max_new_tokens=5)
    out_path = classifier.get_out_path(args.out_dir)
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        tasks = [task for task in set(classifier.data.get_tasks_list()) if task not in existing]
        print(f"{len(existing)} already in file, {len(tasks)} to go")
    else:
        existing = {}
        tasks = list(set(classifier.data.get_tasks_list()))

    all_generated = {}
    for i in range(0, len(tasks), args.tasks_batch_size):
        batch = tasks[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(classifier, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **process_results(all_generated)}
        classifier.dump_results(args.out_dir, all_results_dict)
