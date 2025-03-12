import os
from argparse import ArgumentParser
from prepare_data.classify_constrained_generation_tasks import ConstrainedGenerationClassificationRITS
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tqdm import tqdm
from prepare_data.classify_constrained_generation_tasks import BaseDataset
from datasets import load_from_disk, load_dataset
from big_models_generations_rits import InitGenerationsRITS
from refine_model_response import PROMPT

class RefineData(BaseDataset):

    def get_orig_model_name(self):
        name_or_path = self.get_name_or_path()
        only_name = name_or_path.split(os.sep)[-1]
        name_index = only_name.find("-0shot")
        only_name = only_name[:name_index]
        return only_name

    def load_data(self, name_or_path):
        with open(name_or_path, 'rt') as f:
            data = json.load(f)
        return data

    def get_tasks_list(self):
        return list(self.data.keys())

    def get_constraints_list(self):
        return []


class RevisionsRITS(InitGenerationsRITS):

    def __init__(self, revise_model, data: RefineData):
        super().__init__(data, revise_model)
        self.orig_model = data.get_orig_model_name()

    def get_name(self):
        return f"revise-one-step"

    def get_out_path(self, out_dir):
        path = os.path.join(out_dir, f"{self.short_model_name}-{self.get_name()}-{self.orig_model}.json")
        print(f"output path at {path}")
        return path


def generate_parallel(obj, tasks):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in tasks:
        ai_response = obj.data[task][-1]["content"]
        all_args[task] = (task, api_key, base_url, model_name, ai_response)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(task, api_key, base_url, model_name, ai_response):
    prompt = PROMPT.format(instruction=task, ai_response=ai_response)
    message = [{'role': 'user', 'content': prompt}]
    client = OpenAI(api_key=api_key, base_url=base_url)

    gen_params = {
        GenParams.MAX_NEW_TOKENS: 1000,
        'temperature': 0,
        'extra_headers': {"RITS_API_KEY": api_key}
    }

    if client.base_url.host == "api.openai.com":
        gen_params["max_completion_tokens"] = gen_params.pop("max_new_tokens", None)
    else:
        gen_params['max_tokens'] = gen_params.pop("max_new_tokens", None)

    completion = client.chat.completions.create(
        messages=message,
        model=model_name,
        **gen_params
    )
    generated_text = completion.choices[0].message.content
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_text}]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data_path", help="path to the responses to refine")
    parser.add_argument("--tasks_batch_size", type=int, default=200, help="number of tasks to run inference on before saving")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    dataset = RefineData(args.data_path)

    generator = RevisionsRITS(args.model, dataset)
    out_path = generator.get_out_path(args.out_dir)

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        tasks = [task for task in set(generator.data.get_tasks_list()) if task not in existing]
        print(f"{len(existing)} already in file, {len(tasks)} to go")
    else:
        existing = {}
        tasks = list(set(generator.data.get_tasks_list()))

    all_generated = {}
    for i in range(0, len(tasks), args.tasks_batch_size):
        batch = tasks[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
