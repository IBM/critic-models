import os
from argparse import ArgumentParser
import json
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tqdm import tqdm

from inference_pipeline.big_models_generations_rits import InitGenerationsRITS, OrigData


class GeneralityClassificationRITS(InitGenerationsRITS):
    prompt_template =  \
        ("You are given a constraint from a generation task. "
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
         "\nConstraint: {}\n\nScore: ")

    def get_name(self):
        return "constraint-generality-classification-via-rits"


class DomainClassificationRITS(InitGenerationsRITS):
    domains = "1. Creative Writing\n2. Chemical Industry\n3. Education\n4. Business\n5. Technology\n6. Healthcare\n7. Marketing\n8. Entertainment\n9. Environmental Science\n10. Psychology\n11. Roleplaying\n12. Science Fiction\n13. Fantasy\n14. Journalism\n15. Law\n16. Finance\n17. Data Analysis\n18. Artificial Intelligence\n19. Language Translation\n20. Gaming"

    prompt_template =  \
        ("You are given a generation task. "
         "Classify the domain of the task into one of the domains listed below. Respond only with the category number."
         "\nDomains:\n"+domains+"\n\nTask: {}\n\nYour response:")

    def get_name(self):
        return "task-domain-classification-via-rits"


class ConstraintClassificationRITS(InitGenerationsRITS):
    def __init__(self, model, data: OrigData, categories):
        super().__init__(model, data)

        self.categories_str = ""
        for j, category_dict in enumerate(categories):
            self.categories_str += f"\n{j}. *{category_dict['name']}*: {category_dict['description']}\nExamples: "
            for example in category_dict['examples']:
                self.categories_str += f"\n - {example}"

        self.prompt_template = (
                "Classify the following constraint from a generation task into one of the categories listed below. "
                "Respond only with the category number. "
                "If the constraint does not fit any of the categories from the list, respond with 'Other:' followed by a suggested title for an appropriate category.\n"
                "Categories:"+self.categories_str+"\n\nConstraint: {}\n\nYour response:")

    def get_name(self):
        return "constraint-classification-single-via-rits"


class ConstraintMultilabelClassificationRITS(ConstraintClassificationRITS):
    def __init__(self, model, data: OrigData, categories):
        super().__init__(model, data, categories)

        self.prompt_template = (
                "Classify the following constraint from a generation task into one (or more) of the categories listed below. "
                "Respond only with the category number(s). "
                "If the constraint fits multiple categories, provide the numbers separated by commas (e.g., '1,3,5'). "
                "If the constraint does not fit any of the categories from the list, respond with 'Other:' followed by a suggested title for an appropriate category.\n"
                "Categories:" + self.categories_str + "\n\nConstraint: {}\n\nYour response:")

    def get_name(self):
        return "constraint-classification-multilabel-via-rits"


def generate_parallel(obj, constraints):
    model_name = obj.model_name
    api_key = obj.get_api_key()
    base_url = obj.get_api_endpoint().format(obj.model_name_for_endpoint)
    prompt_template = obj.prompt_template
    all_results = {}
    all_args = {}
    pool = Pool(cpu_count())
    total = 0
    for task in constraints:
        all_args[task] = (task, api_key, base_url, model_name, prompt_template)
        total += 1
    pbar = tqdm(total=total)
    for task, arguments in all_args.items():
        all_results[task] = pool.apply_async(infer_local, arguments, callback=lambda _: pbar.update(1))
    pool.close()
    pool.join()
    print("DONE")
    return {task: task_result.get() for task, task_result in all_results.items()}


def infer_local(constraint, api_key, base_url, model_name, prompt_template):
    msg = prompt_template.format(constraint)
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
    parser.add_argument("--classification_type", type=str, required=True, choices=['generality', 'domain', "constraints_single", "constraints_multilabel"])
    parser.add_argument("--categories_file", type=str)

    args = parser.parse_args()

    dataset = OrigData(args.data_path, args.split, args.tasks_key)

    if args.classification_type == "domain":
        generator = DomainClassificationRITS(args.model, dataset)
    elif args.classification_type == "generality":
        generator = GeneralityClassificationRITS(args.model, dataset)
    else:
        with open(args.categories_file) as f:
            categories = json.load(f)

        if args.classification_type == "constraints_single":
            generator = ConstraintClassificationRITS(args.model, dataset, categories)
        elif args.classification_type == "constraints_multilabel":
            generator = ConstraintMultilabelClassificationRITS(args.model, dataset, categories)

    out_path = generator.get_out_path(args.out_dir)

    tasks_or_constraints = generator.data.get_tasks_list() if args.classification_type == "domain" \
        else generator.data.get_constraints_list()

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        constraints = [con for con in set(tasks_or_constraints) if con not in existing]
        print(f"{len(existing)} already in file, {len(constraints)} to go")
    else:
        existing = {}
        constraints = list(set(tasks_or_constraints))

    all_generated = {}
    for i in range(0, len(constraints), args.tasks_batch_size):
        batch = constraints[i: i + args.tasks_batch_size]
        batch_generated = generate_parallel(generator, batch)
        all_generated = {**all_generated, **batch_generated}
        all_results_dict = {**existing, **all_generated}
        generator.dump_results(args.out_dir, all_results_dict)
