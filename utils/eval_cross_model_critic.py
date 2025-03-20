import os
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd

PREFIX = "decomposition-evaluation-llama3.1-70b."
SEPERATOR = "-revise-one-step-"
SUFFIX = ".json"


def calc_results(path_to_results):
    with open(path_to_results, "r") as f:
        data = json.load(f)
    mean_scores = []
    score_by_sample = {}
    for sample in data:
        all_scores = data[sample]["scores"].values()
        binary_scores = [0 if score == 'ERR' or score < 0.5 else 1 for score in all_scores]
        mean_score = np.mean(binary_scores)
        mean_scores.append(mean_score)
        score_by_sample[sample] = mean_score
    return np.mean(mean_scores), score_by_sample

def generate_labels_for_gen(scores_by_sample):
    group_by_revise_model = scores_by_sample.groupby(["revise_model", "sample"])
    final_labels = []
    for group in group_by_revise_model:
        revise_model, sample = group[0]
        best_rev_score = group[1]["score"].max()
        all_best_models = group[1][group[1]["score"] == best_rev_score]["generator_model"].values
        final_labels.append({
            "revise_model": revise_model,
            "sample": sample,
            "best_gen_models": all_best_models,
            "best_rev_score": best_rev_score
        })
    return final_labels


def generate_labels_for_revision(scores_by_sample):
    group_by_generator_model = scores_by_sample.groupby(["generator_model", "sample"])
    final_labels = []
    for group in group_by_generator_model:
        generator_model, sample = group[0]
        best_rev_score = group[1]["score"].max()
        all_best_models = group[1][group[1]["score"] == best_rev_score]["revise_model"].values
        final_labels.append({
            "generator_model": generator_model,
            "sample": sample,
            "best_rev_models": all_best_models,
            "best_rev_score": best_rev_score
        })
    return final_labels


def eval_dir(dir_path):
    mean_scores = []
    scores_by_sample = []
    for fname in os.listdir(dir_path):
        if not fname.endswith(SUFFIX):
            continue
        only_models = fname.replace(PREFIX, "").replace(SUFFIX, "")
        models_sep = only_models.split(SEPERATOR)
        revise_model = models_sep[0]
        generator_model = models_sep[1]
        mean_score, score_by_sample = calc_results(os.path.join(dir_path, fname))
        mean_scores.append({
            "revise_model": revise_model,
            "generator_model": generator_model,
            "mean_score": mean_score
        })
        all_rows = [{"revise_model": revise_model, "generator_model": generator_model, "sample": sample, "score": score} for sample, score in score_by_sample.items()]
        scores_by_sample.extend(all_rows)

    scores_by_sample_df = pd.DataFrame(scores_by_sample)
    scores_by_sample_df.to_csv(os.path.join(dir_path, "scores_by_sample.csv"))

    labels_generator = generate_labels_for_gen(scores_by_sample_df)
    labels_generator_df = pd.DataFrame(labels_generator)
    labels_generator_df.to_csv(os.path.join(dir_path, "labels_generator.csv"))

    labels_revision = generate_labels_for_revision(scores_by_sample_df)
    labels_revision_df = pd.DataFrame(labels_revision)
    labels_revision_df.to_csv(os.path.join(dir_path, "labels_revision.csv"))

    revision_routing_scores = labels_revision_df.groupby("generator_model")["best_rev_score"].mean()
    mean_scores.extend([{"revise_model": "revision_routing", "generator_model": model, "mean_score": score } for model, score in revision_routing_scores.items()])

    generator_routing_scores = labels_generator_df.groupby("revise_model")["best_rev_score"].mean()
    mean_scores.extend([{"revise_model": model, "generator_model": "generator_routing", "mean_score": score } for model, score in generator_routing_scores.items()])

    mean_scores_df = pd.DataFrame(mean_scores)
    # sort by generator model, then by revise model
    mean_scores_df = mean_scores_df.sort_values(by=["revise_model", "mean_score"], ascending=[True, False])
    # save results
    mean_scores_df.to_csv(os.path.join(dir_path, "mean_scores.csv"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    eval_dir(args.results_dir)

