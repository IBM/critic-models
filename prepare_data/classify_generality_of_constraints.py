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
        return "constraint-generality-classification-via-rits"


def generate_parallel(obj, constraints):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in constraints:
        all_args[task] = (task, api_key, base_url, model_name)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(constraint, api_key, base_url, model_name):
    msg = ("You are given a constraint from a generation task. "
           "Classify the generality of the constraint on a scale from 1 to 5, where 1 is the most general and 5 is the most specific. "
           "Provide your score using the format of [[rating]], for example: '[[3]]'. "
           "General constraints are constraints that can be combined with almost any generation task. "
           "In contrast, specific constraints can only be applied to quite particular situations and requests. "
           "Examples:\n"
           '- Constraint: "Keep the text short and concise." Score: [[1]] Explanation: This constraint is very general and can be added to almost any user request.\n'
           '- Constraint: "The target audience is non-financially aware non-reader young adults." Score: [[3]] Explanation: This is somewhat specific, but can still apply to different types of user requests.\n'
           '- Constraint: "Mention the company "Coca Cola"." Score: [[2]] Explanation: This constraint can in principle be added to a wide array of generative tasks.\n'
           '- Constraint: "Never come across as sounding redundant or repeating yourself." Score: [[1]] Explanation: This is a general guideline to the AI and is not task-specific.\n'
           '''- Constraint: "Describe the main character's desire for independence and his perception of himself as his own man." Score: [[3]] Explanation: This is a constraint that is only relevant for stories, but can apply to many story generation tasks.\n'''           
           '- Constraint: "The hypothesis should be brand-new and not previously proposed." Score: [[4]] Explanation: This constraint will only be applicable to tasks where the assistant is asked to generate a hypothesis.\n'
           '- Constraint: "Explore the possibility of natural hybridization within the genus Sinocyclocheilus." Score: [[5]] Explanation: This is a very specific guideline that appears tied to a particular task.\n'
           '- Constraint: "The output should be in a well-structured JSON format with well-named keys." Score: [[2]] Explanation: The guideline is rather general, but not all tasks can adhere to this desired output format.\n'
           f"\nConstraint: {constraint}\n\nScore: ")
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

    args = parser.parse_args()

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
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)

# pd.DataFrame([{"constraint": k, "score": v.split("[[")[1].split("]]")[0] if "[[" in v else "N/A"} for k, v in json.load(open("output/constraint-generality-classification-via-rits-llama3.3-70b.wild-if-eval.json")).items()]).to_csv("generality.csv", index=False)
