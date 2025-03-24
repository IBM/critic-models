
import glob
from argparse import ArgumentParser
from domain_analysis import  InitGenerationsRITS,infer_local,generate_parallel
from prepare_data.classify_constrained_generation_tasks import BaseDataset
from datasets import load_dataset

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


if __name__ == '__main__':
    import pandas as pd
    import os
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data_path", help="path to the responses to evaluate")
    parser.add_argument("--split", required=True, choices=['train', 'validation', 'test'])
    parser.add_argument("--tasks_key", required=True, help="the tasks column name")
    parser.add_argument("--num_tasks_in_prompt", type=int, default=100)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prompt_batch_size", type=int, default=10,
                        help="number of prompt to run inference on before saving")
    args = parser.parse_args()
    dataset = OrigData(args.data_path, args.split, args.tasks_key)

    generator = InitGenerationsRITS(args.model, dataset, f"domain-analysis-via-rits")
    out_path = generator.get_out_path(args.out_dir)
    df = pd.read_csv('/Users/liate/PycharmProjects/critic-models/_output/scores/for_spss.csv')

    groups = list(set(df['category']))
    prompt_list = []
    print(groups)
    j=0
    for group in groups:
        dff = df[df['category'] == group]
        df_pos = dff[dff['score'] > 0.5]
        df_neg = dff[dff['score'] <= 0.5]
        pos_constraints = df_pos['constraint']
        neg_constraints = df_neg['constraint']
        j=j+1
        for i in range(0, len(pos_constraints), args.prompt_batch_size):
            pos_batch = pos_constraints[i: i + args.prompt_batch_size]
            neg_batch = neg_constraints[i: i + args.prompt_batch_size]
            pos_con_list_str = "\n".join(
                [f"constraint #{i + 1}. {constraint}" for i, constraint in enumerate(pos_batch)])
            neg_con_list_str = "\n".join(
                [f"constraint #{i + 1}. {constraint}" for i, constraint in enumerate(neg_batch)])
            prompt = f"Below are two lists of constrains of type {group}, POS_LIST and NEG_LIST." \
                     f"Find the most prominent difference between the two populations of constraints appearing in the lists. " \
                     f"Describe this difference concisely. " \
                     f"Before mentioning the difference, mention the type of constraints you analyzed. \n\n" \
                     f"POS_LIST:\n {pos_con_list_str}\n" \
                     f"NEG_LIST:\n {neg_con_list_str}\n"
            prompt_list.append(prompt)
        for i in range(0, len(prompt_list), args.prompt_batch_size):
            batch = prompt_list[i: i + args.prompt_batch_size]
            batch_generated = generate_parallel(generator, batch, j, i)
            all_generated = {**all_generated, **batch_generated}
            generator.dump_results(args.out_dir, all_generated)

        pos_con_list_str = "\n".join([f"constraint #{i+1}. {constraint}" for i, constraint in enumerate(pos_constraints) if i<100])
        neg_con_list_str = "\n".join([f"constraint #{i+1}. {constraint}" for i, constraint in enumerate(neg_constraints) if i<100])
        prompt = f"Blow are two lists of constrains of type {group}, POS_LIST and NEG_LIST." \
                 f"Find the most prominent difference between the two populations of constraints appearing in the lists. " \
                 f"Describe this difference concisely. " \
                 f"Before mentioning the difference, mention the type of constraints you analyzed. \n\n" \
                 f"POS_LIST:\n {pos_con_list_str}\n" \
                 f"NEG_LIST:\n {neg_con_list_str}\n"
        prompt_list.append(prompt)

    all_generated = {}
    batch = prompt_list
    batch_generated = generate_parallel(generator, batch, 0)
    all_generated = {**all_generated, **batch_generated}
    generator.set_name(f"constraint-score-analysis")
    generator.dump_results(args.out_dir, all_generated)
    import json
    if os.path.exists('/Users/liate/PycharmProjects/critic-models/_output/constraint-score-analysis-llama3.1-70b.wild-if-eval.json'):
        existing = json.load(open('/Users/liate/PycharmProjects/critic-models/_output/constraint-score-analysis-llama3.1-70b.wild-if-eval.json'))

    for item in existing:
        print(existing[item])

