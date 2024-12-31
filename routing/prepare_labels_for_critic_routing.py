import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import re
"""
For each sample, find which is the best critic to route to.

for each sample, generate a matrix of score after critic, while last row/column will be the initial socre.
row is the generator model, column is the critic model.

"""

SPLIT = 'train'


def map_model_names_to_doc_name(config, generator_model_name, critic_model_name=None):
    if critic_model_name is None:
        path = config["init_path"].format(init_dir=config["init_dir"], eval_prefix=config["evals_prefix"],
                                          generator_model=generator_model_name, split=SPLIT)
    else:
        path = config["revised_path"].format(revised_dir=config["revised_dir"], eval_prefix=config["evals_prefix"],
                                             generator_model=generator_model_name, critic_model=critic_model_name,
                                             split=SPLIT)
    return path


def get_scores_for_all_samples(config):
    """
    For each sample, get the scores for each model.
    """
    models_list = config["generator_models"]
    all_rows = []
    for generator_model in models_list:
        before_revisions_doc = map_model_names_to_doc_name(config, generator_model)
        with open(before_revisions_doc, 'r') as f:
            json_data_before_revision = json.load(f)

        for critic_model in models_list:
            doc_name = map_model_names_to_doc_name(config, generator_model, critic_model)
            with open(doc_name, 'r') as f:
                json_data = json.load(f)
            for sample in tqdm(json_data):
                current_row = {"sample_text": sample, "generator_model": generator_model, "critic_model": critic_model}
                revised_scores = list(json_data[sample]["scores"].values())
                revised_scores = [0 if x == 'ERR' else x for x in revised_scores]  # replace 'ERR' with 0
                mean_constraints_score = np.mean(revised_scores).item()
                current_row["mean_constraints_score_after_revision"] = mean_constraints_score
                current_row["num_constraints"] = len(json_data[sample]["scores"])
                initial_scores = list(json_data_before_revision[sample]["scores"].values())
                initial_scores = [0 if x == 'ERR' else x for x in initial_scores]  # replace 'ERR' with 0
                current_row["mean_constraints_score_before_revision"] = np.mean(initial_scores).item()
                if current_row["num_constraints"] == 0:
                    continue
                all_rows.append(current_row)
    df = pd.DataFrame(all_rows)
    return df


def generate_data_for_training(df):
    """
    Generate the data for training the critic routing model.
    """
    # For each sample, find the critic model that has the highest score.
    group_by_text = df.groupby('sample_text')
    sorted_models = sorted(df["generator_model"].unique().tolist())
    all_rows = []
    for sample_text, group in group_by_text:
        current_row = {"sample_text": sample_text}
        if len(group) < len(sorted_models) * len(sorted_models):
            continue
        scores = []
        for generator_model in sorted_models:
            model_scores = []
            for critic_model in sorted_models:
                model_scores.append(group[(group["generator_model"] == generator_model) & (group["critic_model"] == critic_model)]["mean_constraints_score_after_revision"].values[0].item())
            model_scores.append(group[group["generator_model"] == generator_model]["mean_constraints_score_before_revision"].values[0].item())
            scores.append(model_scores)
        all_scores = np.array(scores)
        current_row["scores"] = all_scores
        all_rows.append(current_row)
    df = pd.DataFrame(all_rows)
    return df


def add_labels(data_df):
    # prepare labels -- best model routing for each sample, is the best critic is self critic?
    self_critic_is_best_all = []
    best_generation_model_all = []
    critic_routing_is_better_than_model_routing_all = []
    no_critics_needed_all = []
    all_best_critics = []
    for i, row in data_df.iterrows():
        scores = row["scores"]
        if type(scores) == str:
            scores = np.array(eval(re.sub(r'\s+', ', ', scores)))
        best_init_generation_score = scores[:, -1].max()
        best_generation_model = scores[:, -1] == best_init_generation_score
        best_generation_model = np.random.choice(np.where(best_generation_model)[0])
        best_generation_model_all.append(best_generation_model)
        revised_scores_for_best_generation_model = scores[best_generation_model]
        best_critic_score = revised_scores_for_best_generation_model.max()
        best_critic_model = revised_scores_for_best_generation_model == best_critic_score
        best_critic_model = np.random.choice(np.where(best_critic_model)[0])
        all_best_critics.append(best_critic_model)
        self_critic_is_best = revised_scores_for_best_generation_model[best_generation_model] == revised_scores_for_best_generation_model.max()
        self_critic_is_best_all.append(self_critic_is_best)
        critic_routing_is_better_than_model_routing = revised_scores_for_best_generation_model.max() < scores.max()
        critic_routing_is_better_than_model_routing_all.append(critic_routing_is_better_than_model_routing)
        no_critic_needed = revised_scores_for_best_generation_model.max() == revised_scores_for_best_generation_model[-1]
        no_critics_needed_all.append(no_critic_needed)
    data_df['best_critic'] = all_best_critics
    data_df['is_self_critic_best'] = self_critic_is_best_all
    data_df['best_init_generation_model'] = best_generation_model_all
    data_df['critic_routing_is_better_than_model_routing'] = critic_routing_is_better_than_model_routing_all
    data_df['no_critics_needed'] = no_critics_needed_all
    return data_df


def main(path_to_config):
    with open(path_to_config, 'r') as f:
        config = json.load(f)
    path_to_df = '_output/data_for_critic_routing.csv'
    if not os.path.exists(path_to_df):
        df = get_scores_for_all_samples(config)
        df = generate_data_for_training(df)
    else:
        df = pd.read_csv(path_to_df)
    df = add_labels(df)
    os.makedirs('_output', exist_ok=True)
    df.to_csv('_output/data_for_critic_routing.csv', index=False)


if __name__ == '__main__':
    main('utils/config_for_routing.json')
