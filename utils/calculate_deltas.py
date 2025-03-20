import os
import json
import numpy as np
import pandas as pd

dir_name = "/Users/gililior/research/py_repos/WildIFEval/llm_aaj_scores"
prefix_0shot = "llm-aaj-llama3.1-70b.{model_name}-0shot-wild-if-eval.json"

dir_revision_scores = "/Users/gililior/research/datasets/wild-if-eval-revisions/llm_aaj/7-9b-models"
prefix_revision = "decomposition-evaluation-llama3.1-70b.{revision_model}-revise-one-step-{generator_model}.json"


def parse_scores(scores, tasks_list):
    mean_scores = []
    for sample in tasks_list:
        task_scores = scores[sample]["scores"]
        binary_scoring = []
        for score in task_scores.values():
            if score == "ERR" or score < 0.5:
                binary_scoring.append(0)
            else:
                binary_scoring.append(1)
        mean_constraint_scores = np.mean(binary_scoring)
        mean_scores.append(mean_constraint_scores)
    return mean_scores

if __name__ == '__main__':
    zero_shot_paths = {model: os.path.join(dir_name, prefix_0shot).format(model_name=model) for model in ["gemma-2-9b", "Llama-3.1-8B", "llama3.3-70b"]}
    revision_paths = {(rev_model, gen_model): os.path.join(dir_revision_scores, prefix_revision).format(revision_model=rev_model, generator_model=gen_model) for rev_model in ["gemma-2-9b", "Llama-3.1-8B"] for gen_model in ["gemma-2-9b", "Llama-3.1-8B"]}

    random_path_from_dict = list(revision_paths.values())[0]
    with open(random_path_from_dict, "r") as f:
        random_scores = json.load(f)
        tasks_list = list(random_scores.keys())
    tasks_list.sort()

    zero_shot_scores = {}
    for model in zero_shot_paths:
        with open(zero_shot_paths[model], "r") as f:
            json_scores = json.load(f)
        zero_shot_scores[model] = parse_scores(json_scores, tasks_list)
    revision_scores = {}
    for models in revision_paths:
        with open(revision_paths[models], "r") as f:
            json_scores = json.load(f)
        revision_scores[models] = parse_scores(json_scores, tasks_list)


    deltas = pd.DataFrame()
    deltas["llama3.1-8b-0shot"] = zero_shot_scores["Llama-3.1-8B"]
    deltas["0shot-gemma-minus-llama"] = np.array(zero_shot_scores["gemma-2-9b"]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["0shot-llama-big-minus-llama-small"] = np.array(zero_shot_scores["llama3.3-70b"]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["self-revision-llama-small-vs-0shot-llama-small"] = np.array(revision_scores[("Llama-3.1-8B", "Llama-3.1-8B")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["gemma-revise-llama-small-vs-0shot-llama-small"] = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["gemma-revise-llama-small-vs-0shot-llama-big"] = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) - np.array(zero_shot_scores["llama3.3-70b"])
    deltas["llama-small-revise-gemma-vs-0shot-llama-small"] = np.array(revision_scores[("Llama-3.1-8B", "gemma-2-9b")]) - np.array(zero_shot_scores["Llama-3.1-8B"])
    deltas["llama-small-revise-gemma-vs-0shot-llama-big"] = np.array(revision_scores[("Llama-3.1-8B", "gemma-2-9b")]) - np.array(zero_shot_scores["llama3.3-70b"])
    deltas.to_csv("_output/deltas.csv", index=False)


    samples_with_small_model_perfect_scoring = np.array(zero_shot_scores["Llama-3.1-8B"])==1
    gemma_revise_llama_doesnt_improve = np.array(revision_scores[("gemma-2-9b", "Llama-3.1-8B")]) <= np.array(zero_shot_scores["Llama-3.1-8B"])
    bigger_llama_no_better_than_small = np.array(zero_shot_scores["llama3.3-70b"]) <= np.array(zero_shot_scores["Llama-3.1-8B"])
    both_revision_and_bigger_not_better = np.logical_and(gemma_revise_llama_doesnt_improve, bigger_llama_no_better_than_small)
    # both_revision_and_bigger_not_better = np.logical_and(both_revision_and_bigger_not_better, ~samples_with_small_model_perfect_scoring)

    revision_with_gemma_is_best = deltas["gemma-revise-llama-small-vs-0shot-llama-big"] >= 0
    revision_with_gemma_is_best = np.logical_and(revision_with_gemma_is_best, ~both_revision_and_bigger_not_better)

    inference_big_llama_0shot_is_best = deltas["gemma-revise-llama-small-vs-0shot-llama-big"] < 0
    inference_big_llama_0shot_is_best = np.logical_and(inference_big_llama_0shot_is_best, ~both_revision_and_bigger_not_better)

    dataset = pd.DataFrame()
    dataset["sample"] = tasks_list
    dataset["small_model_perfect"] = samples_with_small_model_perfect_scoring
    dataset["both_revision_and_bigger_not_better"] = both_revision_and_bigger_not_better
    dataset["revision_with_gemma_is_best"] = revision_with_gemma_is_best
    dataset["inference_big_llama_0shot_is_best"] = inference_big_llama_0shot_is_best
    dataset.to_csv("_output/dataset.csv", index=False)

    for col in dataset.columns[1:]:
        print(f"{col}: {dataset[col].sum()/len(dataset)}")




