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


class OrigData(BaseDataset):

    def __init__(self, name_or_path, split, tasks_key):
        self.split = split
        self.tasks_key = tasks_key
        super().__init__(name_or_path)

    def get_name_or_path(self):
        return self.name_or_path

    def load_data(self, name_or_path):
        data = load_dataset(name_or_path, split=self.split)
        return data

    def get_tasks_list(self):
        return list(self.data[self.tasks_key])

    def get_constraints_list(self):
        all_constraints = [item for sublist in self.data["decomposition"] for item in sublist]
        all_unique_constraints = list(set(all_constraints))
        return all_unique_constraints


class InitGenerationsRITS(ConstrainedGenerationClassificationRITS):

    def __init__(self, model, data: OrigData):
        super().__init__(data, model, max_new_tokens=1000)

    def get_name(self):
        return f"init-generations-via-rits"

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


def generate_parallel(obj, prompt_lists):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0

    for prompt_list in prompt_lists:
        all_args[prompt_list] = (prompt_list, api_key, base_url, model_name)
        total += 1
    pbar = tqdm(total=total)
    for prompt_list, arguments in all_args.items():
        all_results[prompt_list] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))

    pool.close()
    pool.join()
    print("DONE")
    return {first_prompt: task_result.get() for first_prompt, task_result in all_results.items()}


def infer_local(task, api_key, base_url, model_name):
    message = [{'role': 'user', 'content': task}]
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
    return [{"role": "user", "content": task}, {"role": "assistant", "content": generated_text}]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data_path", help="path to the responses to evaluate")
    parser.add_argument("--split", required=True, choices=['train', 'validation', 'test'])
    parser.add_argument("--tasks_key", required=True, help="the tasks column name")
    parser.add_argument("--tasks_batch_size", type=int, default=200,
                        help="number of prompt batches to run inference on before saving")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prompt_batch_size", type=int, default=5,
                        help="number of prompt to run in a single inference cycle")
    args = parser.parse_args()
    dataset = OrigData(args.data_path, args.split, args.tasks_key)

    generator = InitGenerationsRITS(args.model, dataset)
    out_path = generator.get_out_path(args.out_dir)

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        tasks = [task for task in set(generator.data.get_tasks_list()) if task not in existing]
        print(f"{len(existing)} already in file, {len(tasks)} to go")
    else:
        existing = {}
        tasks = list(set(generator.data.get_tasks_list()))

    prompt_lists = []
    for i in range(0, len(tasks), args.tasks_batch_size):
        batch = tasks[i: i + args.tasks_batch_size]
        task_list = "\n".join([f"Task #{i+1}. {task}" for i, task in enumerate(batch)])
        prompt_lists.append(f"Each of the following tasks can be associated with a specific domain. " 
                     "Generate a list of 20 domains that best represent the domains associated with "
                       f"the tasks.\n Here is the list of tasks:\n\n{task_list}")

    all_generated = {}
    for i in range(0, len(prompt_lists), args.prompt_batch_size):
        batch = prompt_lists[i: i + args.prompt_batch_size]
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
