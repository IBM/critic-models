import os
from argparse import ArgumentParser
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tqdm import tqdm

from inference_pipeline.big_models_generations_rits import InitGenerationsRITS, OrigData


class ClassificationRITS(InitGenerationsRITS):
    def get_name(self):
        return "constraint-classification-via-rits"


def generate_parallel(obj, constraints, all_categories):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in constraints:
        all_args[task] = (task, all_categories, api_key, base_url, model_name)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(constraint, all_categories: list[dict], api_key, base_url, model_name):
    str_categories = ""
    for j, category_dict in enumerate(all_categories):
        str_categories += f"\n{j}. *{category_dict['name']}*: {category_dict['description']}\nExamples: "
        for example in category_dict['examples']:
            str_categories += f"\n - {example}"
    msg = f"Classify the following constraint from a generation task into one (or more) of the categories listed below. Respond only with the category number(s). If the constraint fits multiple categories, provide the numbers separated by commas (e.g., '1,3,5'). If the constraint does not fit any of the categories from the list, respond with 'Other:' followed by a suggested title for an appropriate category.\nCategories:{str_categories}\n\nConstraint: {constraint}\n\nYour response:"
    message = [{'role': 'user', 'content': msg}]
    client = OpenAI(api_key=api_key, base_url=base_url)

    gen_params = {
        GenParams.MAX_NEW_TOKENS: 10,
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
    return generated_text


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tasks_key", required=True, help="the tasks column name")
    parser.add_argument("--tasks_batch_size", type=int, default=200, help="number of tasks to run inference on before saving")
    parser.add_argument("--categories_file", type=str, required=True)

    args = parser.parse_args()

    with open(args.categories_file) as f:
        categories = json.load(f)

    dataset = OrigData(args.data_path, args.split, args.tasks_key)

    generator = ClassificationRITS(args.model, dataset)
    out_path = generator.get_out_path(args.out_dir)

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        constraints = [con for con in set(generator.data.get_constraints_list()) if con not in existing]
        print(f"{len(existing)} already in file, {len(constraints)} to go")
    else:
        existing = {}
        constraints = list(set(generator.data.get_constraints_list()))

    all_generated = {}
    for i in range(0, len(constraints), args.tasks_batch_size):
        batch = constraints[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(generator, batch, categories)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
